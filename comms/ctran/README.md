# CTRAN

CTRAN emerged as a solution to challenges in NCCL, providing a modular, self-contained architecture for collective communications across different GPU types (NVIDIA, AMD) and network topologies.

## Development Guidelines

- **Boyscout Rule:**
  Ctran has evolved over time, and the codebase may not always be clean. The following rules are designed to keep the codebase maintainable, but you may encounter legacy code that breaks these rules. If you find issues, please fix them and strive to keep the code clean.

- **Ctran is an independent library:**
  Do not add dependencies on nccl, ncclx, rcclx, or mccl.

- **Usage Scenarios:**
  Ctran is used in various environments, including NV and AMD GPUs. Test thoroughly and avoid making assumptions about usage.

- **Submodule Dependencies:**
  You may use small submodules or utilities from the comms library, but do not depend on anything that introduces nccl, ncclx, rcclx, or mccl.

- **Modularity:**
  Build modular and independent libraries. Avoid large, monolithic repositories.

- **Minimal Dependencies:**
  Keep dependencies minimal. Prefer using `stateX` instead of `CtranComm` when possible.

- **Testing:**
  Every new code must have tests.

- **C++ Best Practices:**
    - Use smart pointers and guards.
    - Do not use `goto` statements.
    - Use Folly if possible.
    - Prefer exceptions over return codes.
