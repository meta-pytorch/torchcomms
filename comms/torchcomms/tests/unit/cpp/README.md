# C++ Unit Tests

## Build and Run all Tests

From the repository root:

```bash
ctest --build-and-test ./ ./build --build-generator "Ninja" --build-options -DBUILD_TESTS=ON --test-command ctest --output-on-failure
```

This will configure, build, and run all tests in one command.

## Building Tests Only

To configure and build without running tests:

```bash
cd build
cmake .. -G Ninja -DBUILD_TESTS=ON
ninja
```

To build a specific test executable:

```bash
ninja TorchCommFactoryTest
ninja TorchCommOptionsTest
```

## Running Tests (after building)

From the build directory:

```bash
cd build
ctest --output-on-failure
```

### Running Individual Tests

Use ctest's `-R` flag for regex matching:

```bash
# Run only TorchCommOptionsTest tests
ctest -R Options --output-on-failure

# Run only TorchCommFactoryTest tests
ctest -R Factory --output-on-failure

# Run a specific test by name
ctest -R CreateGenericBackend --output-on-failure
```

### Rebuilding and Rerunning a Specific Test

After modifying test code, rebuild and rerun a specific test:

```bash
cd build

# Rebuild a specific test executable
ninja TorchCommFactoryTest

# Run it
ctest -R Factory --output-on-failure
```
