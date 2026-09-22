# AMD_SMI Component

The **AMD_SMI** (AMD System Management Interface) component exposes hardware
management counters (and selected controls) for AMD GPUs — e.g., power usage,
temperatures, clocks, PCIe link metrics, VRAM information, and RAS/ECC status —
by querying the AMD SMI library at runtime (ROCm ≥ 6.4.0).

> **Configure note.** When both `amd_smi` and `rocm_smi` are requested,
> PAPI’s configure script now inspects the ROCm stack and enables only the
> appropriate SMI backend. We select `amd_smi` for ROCm 6.4.0 and newer, and
> keep `rocm_smi` for older releases. This cutoff is based on internal testing
> that showed AMD SMI becoming stable and feature-complete beginning with ROCm
> 6.4.0.

- [Environment Variables](#environment-variables)
- [Enabling the AMD_SMI Component](#enabling-the-amd_smi-component)
- [Hardware and Software Support](#hardware-and-software-support)

---

## Environment Variables

For AMD_SMI, PAPI requires the environment variable `PAPI_AMDSMI_ROOT` to be set
so that the AMD SMI shared library and headers can be found. This variable is
required at both **compile** and **run** time.

**Setting PAPI_AMDSMI_ROOT**  
Set `PAPI_AMDSMI_ROOT` to the top-level ROCm directory. For example:

   ```bash
   export PAPI_AMDSMI_ROOT=/opt/rocm-6.4.0
   # or
   export PAPI_AMDSMI_ROOT=/opt/rocm
   ```

The directory specified by `PAPI_AMDSMI_ROOT` **must contain** the following
subdirectories:

- `PAPI_AMDSMI_ROOT/lib` (which should include the dynamic library `libamd_smi.so`)
- `PAPI_AMDSMI_ROOT/include/amd_smi` (AMD SMI headers)

If the library is not found or is not functional at runtime, the component will
appear as "disabled" in `papi_component_avail`, with a message describing the
problem (e.g., library not found).

### Library search order

At initialization the component constructs the full path
`${PAPI_AMDSMI_ROOT}/lib/libamd_smi.so` and hands it to `dlopen(3)`. If the file
is missing or unreadable the component is disabled immediately. Any additional
dependencies that `libamd_smi.so` brings in are resolved by the platform loader
using the standard order:

- entries in `LD_LIBRARY_PATH`
- rpaths encoded in the binary or library
- system defaults such as `/etc/ld.so.conf`, `/usr/lib64`, `/lib64`, `/usr/lib`,
  and `/lib`

Because the main shared object is loaded by absolute path, pointing
`PAPI_AMDSMI_ROOT` at the directory tree that actually contains AMD SMI is the
authoritative way to pick up non-standard installs. Symlinking
`${PAPI_AMDSMI_ROOT}/lib/libamd_smi.so` to the desired copy also works.

### Handling non-standard installations

- **Modules or package managers** – environment modules (`module load rocm`),
  Spack, or distro packages typically extend `PATH`, `LD_LIBRARY_PATH`, and
  other variables for you. Set `PAPI_AMDSMI_ROOT` to the corresponding ROCm
  prefix exported by the tool (check with `printenv` or `spack location`).
- **Bare installs** – if AMD SMI lives elsewhere, export
  `PAPI_AMDSMI_ROOT=/custom/prefix` so that `${PAPI_AMDSMI_ROOT}/lib` and
  `${PAPI_AMDSMI_ROOT}/include` resolve correctly.
- **Dependent libraries** – when a vendor build puts required runtime libraries
  (e.g., HIP, ROCm math libs) outside the ROCm tree, append those directories to
  `LD_LIBRARY_PATH`, for example:

  ```bash
  export LD_LIBRARY_PATH="/usr/lib64:/opt/vendor-extra/lib:${LD_LIBRARY_PATH}"
  ```

  Always append/prepend to the existing variable to avoid clobbering entries
  added by other packages.

---

## Enabling the AMD_SMI Component

To enable reading (and where supported, writing) of AMD_SMI counters, build
PAPI with this component enabled. For example:

```bash
./configure --with-components="amd_smi"
make
```

You can verify availability with the utilities in `papi/src/utils/`:

```bash
papi_component_avail            # shows enabled/disabled components
papi_native_avail -i amd_smi    # lists native events for this component
```

After changing `PAPI_AMDSMI_ROOT` or related library paths, rerun make clobber && ./configure --with-components="amd_smi" before rebuilding so configure picks up the new locations.

---


## Hardware and Software Support
To see the `amd_smi` component's current supported hardware and software please visit the GitHub wiki page [Hardware and Software Support - AMD\_SMI Component](https://github.com/icl-utk-edu/papi/wiki/Hardware-and-Software-Support-%E2%80%90-AMD_SMI-Component).
