As of now, the GitHub CI is designed to run in three instances:
1. An individual component basis, meaning if a component's codebase is updated, we will only run CI tests for that component. As an example, if we update `cupti_profiler.c` in `src/components/cuda`, we will only run CI tests for the cuda component. Note that this includes updates to subdirectories located in a component's directory (e.g. `src/components/cuda/tests`). See the **NOTE** in [Individual Component Basis](#individual-component-basis) for more info on the setup for the default components (`perf_event`, `perf_event_uncore`, and `sysdetect`).
2. A change to the Counter Analysis Toolkit i.e. in the `src/counter_analysis_toolkit` directory and any subdirectories.
3. A change in the PAPI framework i.e. in the `src/` directory or `src/` subdirectories (excluding individual components and the Counter Analysis Toolkit). If this occurs then we will run a full test suite.


# Individual Component Basis
All individual component basis tests have a `.yml` that is structured with `componentName_component_workflow.yml`. As 
an example for the `cuda` component we would have a `.yml` of `cuda_component_workflow.yml`. Therefore,
if a new component is added to PAPI, you will need to create a `.yml` based on the aforementioned structure.

Upon creating the `.yml` file, you will need to add a workflow. Below is a skeleton that can
be used as a starting point. As a reminder, make sure to change the necessary fields out for your component.

```yml
name: cuda # replace cuda with your component name

on:
  pull_request:
    paths:
      - 'src/components/cuda/**' # replace the cuda path with your component path

jobs:
  component_tests:
    strategy:
      matrix:
        component: [cuda] # replace cuda with your component name
        debug: [yes, no] 
        shlib: [with, without]
      fail-fast: false
    runs-on: [self-hosted, nvidia_gpu]
    timeout-minutes: 60
    steps:
      - uses: actions/checkout@v4
      - name: cuda component tests # replace cuda with your component name
        run: .github/workflows/ci_individual_component.sh ${{matrix.component}} ${{matrix.debug}} ${{matrix.shlib}}
```` 

For each component `.yml`, there will be a single job with the options of:
  - `component`, this is the component we want to configure PAPI with (e.g. cuda)
  - `debug`, with the options of either `yes` (builds a debug version) or `no` (does not build a debug version)
  - `shlib`, with `--with-shlib-tools` or without `--with-shlib-tools`

These options will be used in the script `ci_individual_component.sh` to test configuring and building PAPI. 

Besides configuring and building PAPI, `ci_individual_component.sh` will:
  - Check to make sure components are active that we expect
  - Run a test suite using either `run_tests.sh` (without `--with-shlib-tools`) or `run_tests_shlib.sh` (with `--with-shlib-tools`)

**NOTE**: The components `perf_event`, `perf_event_uncore`, and `sysdetect` do not follow the above outlined file structure. 
For these three components the files used are `default_components_workflow.yml` and `ci_default_components.sh`. Even though
the file structure is different the workflow will still only run if a change is made to one of their directories or subdirectories.
The reason for this is that these three components are compiled in by default and trying to pass one of them to `--with-components=`
will result in an error during the build process. Therefore, any PAPI CI updates for one of these three components would need to be
addressed in either of the two aforementioned files.

# Counter Analysis Toolkit
The Counter Analysis Toolkit (CAT) CI uses the `cat_workflow.yml` and `ci_cat.sh` files. Any updates to the CI for CAT need to be done in these two files.

The `cat_workflow.yml` will have a single job with the options of:
  - `debug`, with the options of either `yes` (builds a debug version) or `no` (does not build a debug version)
  - `shlib`, with `--with-shlib-tools` or without `--with-shlib-tools`

These options will be used in the script `ci_cat.sh` to test configuring and building PAPI.

Besides configuring and building PAPI `ci_cat.sh` will:
- Test building CAT
- Check to see if CAT successfully detects the architecture we are on
- Run CAT with a real event and a dummy event
  - For the real event, we expect the file to exist and values to be present
  - For the dummy event, we expect the file to exist, but values to not be present

# PAPI Framework
The PAPI framework CI uses the `papi_framework_workflow.yml` along with the scripts `clang_analysis.sh`, `ci_papi_framework.sh`, and `spack.sh`. Any updates to the CI for the PAPI framework need to be done in these files.

`papi_framework_workflow.yml` will have a total of five jobs:
1. papi_components_comprehensive
   - With the options of:
     - `components`, this is the components we want to configure PAPI with, i.e. `cuda nvml rocm rocm_smi powercap powercap_ppc rapl sensors_ppc infiniband net appio io lustre stealtime`
     - `debug`, with the options of either `yes` (builds a debug version) or `no` (does not build a debug version)
     - `shlib`, with `--with-shlib-tools` or without `--with-shlib-tools`
    
2. papi_components_amd
   - With the options of:
     - `components`, this is the components we want to configure PAPI with, i.e. `rocm rocm_smi`
     - `debug`, with the options of either `yes` (builds a debug version) or `no` (does not build a debug version)
     - `shlib`, with `--with-shlib-tools` or without `--with-shlib-tools`

3. papi_component_infiniband
   - With the options of:
     - `component`, this is the component we want to configure PAPI with, i.e. `infiniband`
     - `debug`, with the options of either `yes` (builds a debug version) or `no` (does not build a debug version)
     - `shlib`, with `--with-shlib-tools` or without `--with-shlib-tools`

4. papi_spack
   - The script `spack.sh` will be ran, which configures and builds PAPI from SPACK

5. papi_clang_analysis
   - The script `clang_analysis.sh` will be ran, which configures and builds PAPI with clang

For jobs 1, 2, and 3, the options listed will be used in the script `ci_papi_framework.sh` to test configuring and building PAPI.

Besides configuring and building PAPI, `ci_papi_framework.sh` will:
  - Check to make sure components are active that we expect
  - Run a test suite using either `run_tests.sh` (without `--with-shlib-tools`) or `run_tests_shlib.sh` (with `--with-shlib-tools`)
