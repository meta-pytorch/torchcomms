# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import itertools
import logging
import os
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendInstance:
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"


@dataclass(frozen=True)
class ProxyConfig:
    host: str
    port: int
    prefiller_instances: list[BackendInstance]
    decoder_instances: list[BackendInstance]


class BackendClient:
    """HTTP client for one prefill or decode vLLM backend."""

    def __init__(self, instance: BackendInstance, backend_id: int) -> None:
        self.instance = instance
        self.backend_id = backend_id
        self._client = httpx.AsyncClient(
            timeout=None,
            base_url=instance.base_url,
            limits=httpx.Limits(
                max_connections=None,
                max_keepalive_connections=None,
            ),
        )

    async def post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> httpx.Response:
        response = await self._client.post(endpoint, json=payload, headers=headers)
        response.raise_for_status()
        await response.aread()
        return response

    async def stream_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> AsyncIterator[bytes]:
        async with self._client.stream(
            "POST",
            endpoint,
            json=payload,
            headers=headers,
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                yield chunk

    async def close(self) -> None:
        await self._client.aclose()

    def __repr__(self) -> str:
        return (
            f"BackendClient(id={self.backend_id}, "
            f"host={self.instance.host!r}, port={self.instance.port})"
        )


class BackendPool:
    """Round-robin pool for one backend role."""

    def __init__(self, role: str, instances: list[BackendInstance]) -> None:
        if not instances:
            raise ValueError(f"{role} backend pool requires at least one instance")
        self._role = role
        self._instances = instances
        self._clients: list[BackendClient] = []
        self._iterator: itertools.cycle[int] | None = None

    @property
    def size(self) -> int:
        return len(self._clients)

    def start(self) -> None:
        self._clients = [
            BackendClient(instance, backend_id)
            for backend_id, instance in enumerate(self._instances)
        ]
        self._iterator = itertools.cycle(range(len(self._clients)))
        logger.info("Initialized %s %s backend clients", self.size, self._role)

    async def close(self) -> None:
        for client in self._clients:
            await client.close()
        self._clients = []
        self._iterator = None

    def next_client(self) -> BackendClient:
        if self._iterator is None:
            raise RuntimeError(f"{self._role} backend pool is not initialized")
        return self._clients[next(self._iterator)]


class CompletionRequestBuilder:
    """Builds the two requests used by the disaggregated prefill/decode flow."""

    @staticmethod
    def headers(request_id: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "X-Request-Id": request_id,
        }

    @staticmethod
    def prefill_payload(request_data: dict[str, Any]) -> dict[str, Any]:
        payload = request_data.copy()
        payload["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        payload["stream"] = False
        payload["max_tokens"] = 1
        if "max_completion_tokens" in payload:
            payload["max_completion_tokens"] = 1
        payload.pop("stream_options", None)
        return payload

    @staticmethod
    def decode_payload(
        request_data: dict[str, Any],
        kv_transfer_params: dict[str, Any],
    ) -> dict[str, Any]:
        payload = request_data.copy()
        if kv_transfer_params:
            payload["kv_transfer_params"] = kv_transfer_params
        return payload


class DisaggProxyService:
    """Coordinates prefill and decode requests for OpenAI-compatible endpoints."""

    def __init__(self, config: ProxyConfig) -> None:
        self._prefill_pool = BackendPool("prefill", config.prefiller_instances)
        self._decode_pool = BackendPool("decode", config.decoder_instances)
        self._builder = CompletionRequestBuilder()

    def start(self) -> None:
        self._prefill_pool.start()
        self._decode_pool.start()

    async def close(self) -> None:
        await self._prefill_pool.close()
        await self._decode_pool.close()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "prefill_instances": self._prefill_pool.size,
            "decode_instances": self._decode_pool.size,
        }

    async def handle_completion(
        self,
        endpoint: str,
        request_data: dict[str, Any],
    ) -> StreamingResponse:
        request_id = str(uuid.uuid4())
        headers = self._builder.headers(request_id)

        prefill_client = self._prefill_pool.next_client()
        prefill_response = await prefill_client.post_json(
            endpoint,
            self._builder.prefill_payload(request_data),
            headers,
        )
        try:
            response_payload = prefill_response.json()
            kv_transfer_params = response_payload.get("kv_transfer_params")
            if not kv_transfer_params:
                raise ValueError("prefill response did not include kv_transfer_params")
        finally:
            await prefill_response.aclose()

        decode_client = self._decode_pool.next_client()
        decode_payload = self._builder.decode_payload(
            request_data,
            kv_transfer_params,
        )
        logger.debug(
            "Routing request %s via %s -> %s", request_id, prefill_client, decode_client
        )

        async def stream_decode_response() -> AsyncIterator[bytes]:
            async for chunk in decode_client.stream_json(
                endpoint,
                decode_payload,
                headers,
            ):
                yield chunk

        return StreamingResponse(
            stream_decode_response(),
            media_type="application/json",
        )


def _get_service(request: Request) -> DisaggProxyService:
    return request.app.state.proxy_service


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    service: DisaggProxyService = app.state.proxy_service
    service.start()
    try:
        yield
    finally:
        await service.close()


def create_app(config: ProxyConfig) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.state.proxy_service = DisaggProxyService(config)
    app.add_api_route("/v1/completions", handle_completions, methods=["POST"])
    app.add_api_route(
        "/v1/chat/completions",
        handle_chat_completions,
        methods=["POST"],
    )
    app.add_api_route("/healthcheck", healthcheck, methods=["GET"])
    return app


def parse_args(argv: list[str] | None = None) -> ProxyConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    parser.add_argument(
        "--prefiller-hosts",
        "--prefiller-host",
        type=str,
        nargs="+",
        default=["localhost"],
    )
    parser.add_argument(
        "--prefiller-ports", "--prefiller-port", type=int, nargs="+", default=[8100]
    )

    parser.add_argument(
        "--decoder-hosts", "--decoder-host", type=str, nargs="+", default=["localhost"]
    )
    parser.add_argument(
        "--decoder-ports", "--decoder-port", type=int, nargs="+", default=[8200]
    )

    args = parser.parse_args(argv)

    if len(args.prefiller_hosts) != len(args.prefiller_ports):
        raise ValueError(
            "Number of prefiller hosts must match number of prefiller ports"
        )

    if len(args.decoder_hosts) != len(args.decoder_ports):
        raise ValueError("Number of decoder hosts must match number of decoder ports")

    return ProxyConfig(
        host=args.host,
        port=args.port,
        prefiller_instances=[
            BackendInstance(host, port)
            for host, port in zip(args.prefiller_hosts, args.prefiller_ports)
        ],
        decoder_instances=[
            BackendInstance(host, port)
            for host, port in zip(args.decoder_hosts, args.decoder_ports)
        ],
    )


async def _handle_completions(endpoint: str, request: Request) -> StreamingResponse:
    try:
        return await _get_service(request).handle_completion(
            endpoint,
            await request.json(),
        )
    except httpx.HTTPError:
        logger.exception("HTTP error in disagg proxy for %s endpoint", endpoint)
        raise
    except (KeyError, ValueError, RuntimeError):
        logger.exception(
            "Request handling error in disagg proxy for %s endpoint",
            endpoint,
        )
        raise


async def handle_completions(request: Request) -> StreamingResponse:
    return await _handle_completions("/completions", request)


async def handle_chat_completions(request: Request) -> StreamingResponse:
    return await _handle_completions("/chat/completions", request)


async def healthcheck(request: Request) -> dict[str, Any]:
    return _get_service(request).health()


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    config = parse_args(argv)

    import uvicorn

    uvicorn.run(create_app(config), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
