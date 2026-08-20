// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Rust wrappers for NCCL's public host API.
//!
//! This crate owns NCCL resources and gives scalar element types a checked
//! mapping to `ncclDataType_t`. Current communication methods accept raw CUDA
//! pointers and are declared `unsafe`; their contracts require valid,
//! CUDA-accessible, correctly sized buffers to remain alive until stream
//! completion. Higher-level buffer wrappers can encode those requirements in
//! safe interfaces.

mod communicator;
mod config;
mod device;
mod error;
mod group;
mod memory;
mod types;

pub use communicator::AsyncStatus;
pub use communicator::Communicator;
pub use communicator::CommunicatorState;
pub use config::Config;
pub use config::CtaPolicy;
pub use config::GraphUsageMode;
pub use device::DeviceCommRequirements;
pub use device::DeviceCommunicator;
pub use device::GinConnectionType;
pub use device::GinType;
pub use error::Error;
pub use error::Result;
pub use error::Status;
pub use group::Group;
pub use memory::NcclMemory;
pub use memory::Window;
pub use memory::WindowFlags;
/// Raw bindings used by this wrapper.
///
/// Prefer the owned types in this crate. This re-export is provided for APIs
/// that have not acquired a Rust wrapper yet.
pub use nccl_sys as sys;
pub use types::BFloat16;
pub use types::CudaStream;
pub use types::Float8E4M3;
pub use types::Float8E5M2;
pub use types::Float16;
pub use types::NcclDataType;
pub use types::ReductionOp;
pub use types::UniqueId;
pub use types::Version;
