# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

"""Adapter interface for Triton communication kernel tuning."""

from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import torch

KeyT = TypeVar("KeyT")
ConfigT = TypeVar("ConfigT")
InputSpecT = TypeVar("InputSpecT")
InputsT = TypeVar("InputsT")
OutputT = TypeVar("OutputT")


class CommKernelTuningAdapter(
    ABC,
    Generic[KeyT, ConfigT, InputSpecT, InputsT, OutputT],
):
    """Base interface for one Triton communication kernel tuner."""

    name: str

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        pass

    def configure(self, args: argparse.Namespace) -> None:
        pass

    @abstractmethod
    def enumerate_input_specs(self, world_size: int) -> list[InputSpecT]:
        raise NotImplementedError

    @abstractmethod
    def make_key(self, spec: InputSpecT, world_size: int) -> KeyT:
        raise NotImplementedError

    @abstractmethod
    def enumerate_candidate_configs(
        self,
        spec: InputSpecT,
        key: KeyT,
    ) -> list[ConfigT]:
        raise NotImplementedError

    def enumerate_baselines(self, spec: InputSpecT, key: KeyT) -> list[str]:
        return []

    @abstractmethod
    def make_inputs(
        self,
        spec: InputSpecT,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> InputsT:
        raise NotImplementedError

    @abstractmethod
    def run_candidate(
        self,
        inputs: InputsT,
        config: ConfigT,
        group: Any,
    ) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def run_baseline(
        self,
        inputs: InputsT,
        baseline: str,
        group: Any,
    ) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def run_reference(self, inputs: InputsT, group: Any) -> OutputT:
        raise NotImplementedError

    @abstractmethod
    def check_correctness(
        self,
        candidate_output: OutputT,
        reference_output: OutputT,
    ) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def spec_to_json(self, spec: InputSpecT) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def spec_from_json(self, spec_type: str, data: dict[str, Any]) -> InputSpecT:
        raise NotImplementedError

    @abstractmethod
    def key_to_json(self, key: KeyT) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def key_from_json(self, key_type: str, data: dict[str, Any]) -> KeyT:
        raise NotImplementedError

    @abstractmethod
    def config_to_json(self, config: ConfigT) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def config_from_json(
        self,
        config_type: str,
        data: dict[str, Any],
    ) -> ConfigT:
        raise NotImplementedError

    @abstractmethod
    def generated_table_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generated_filename(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def generated_imports(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_key(self, key_type: str, key: dict[str, Any]) -> str:
        raise NotImplementedError

    @abstractmethod
    def render_config(self, config_type: str, config: dict[str, Any]) -> str:
        raise NotImplementedError

    def render_result_comment(self, result: dict[str, Any]) -> str | None:
        return None

    def result_is_artifact_eligible(self, result: dict[str, Any]) -> bool:
        return True
