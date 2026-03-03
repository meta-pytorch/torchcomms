// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "comms/torchcomms/TorchCommFactory.hpp"

#include <dlfcn.h>

#include "comms/torchcomms/TorchComm.hpp"
#include "comms/torchcomms/utils/Logging.hpp"

namespace torch::comms {

namespace {

// BackendLib manages the lifecycle of dynamically loaded backend libraries.
// The library is loaded in the constructor via dlopen() and unloaded in the
// destructor via dlclose(). Move semantics transfer ownership of the library
// handle, while copy operations are explicitly deleted to prevent dangling
// references. After loading, setLoader() must be called to initialize the
// DynamicLoaderInterface which provides the new_comm/destroy_comm functions
// for creating backend instances.
class BackendLib {
 public:
  explicit BackendLib(const char* path) {
    TC_LOG(INFO) << "Loading backend library: " << path;
    handle_ = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
    if (handle_ == nullptr) {
      std::stringstream error_msg;
      error_msg << "Failed to load backend library: " << dlerror()
                << " path=" << path;
      throw std::runtime_error(error_msg.str());
    }
  }

  BackendLib(BackendLib&& other) noexcept
      : handle_(other.handle_), loader_(other.loader_) {
    other.handle_ = nullptr;
  }

  ~BackendLib() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  // Non-copyable to avoid dangling references to the handle
  BackendLib(const BackendLib&) = delete;
  BackendLib& operator=(const BackendLib&) = delete;
  BackendLib& operator=(BackendLib&&) = delete;

  bool setLoader(const std::string& loader_fn_name, std::stringstream& err) {
    // Load the create_dynamic_loader function
    auto create_loader_fn = reinterpret_cast<CreateDynamicLoaderFn>(
        dlsym(handle_, loader_fn_name.c_str()));

    if (create_loader_fn == nullptr) {
      err << "Failed to load function " << loader_fn_name
          << " from backend library";
      return false;
    }

    // Get the dynamic loader interface
    loader_ = create_loader_fn();

    // Validate that all required function pointers are present
    if (loader_.new_comm == nullptr || loader_.destroy_comm == nullptr ||
        loader_.get_supported_version == nullptr) {
      err << "Dynamic loader interface missing required function pointers";
      return false;
    }

    // Check version compatibility
    std::string supported_version = loader_.get_supported_version();
    if (supported_version != TORCHCOMM_BACKEND_ABI_VERSION) {
      err << "ABI version mismatch: " << supported_version
          << " != " << TORCHCOMM_BACKEND_ABI_VERSION;
      return false;
    }

    return true;
  }
  DynamicLoaderInterface getLoader() {
    return loader_;
  }

 private:
  void* handle_{nullptr};
  DynamicLoaderInterface loader_{};
};

BackendLib getBackendLib(const std::string& backend) {
  // Look for environment variable that tells us where to load the backend
  std::string env_key = "TORCHCOMMS_BACKEND_LIB_PATH_" + backend;
  std::transform(
      env_key.begin(), env_key.end(), env_key.begin(), [](unsigned char c) {
        return std::toupper(c);
      });

  const char* backend_lib_path = std::getenv(env_key.c_str());
  if (backend_lib_path == nullptr) {
    std::stringstream error_msg;
    error_msg << "Backend " << backend << " specified, but " << env_key
              << " not set";
    throw std::runtime_error(error_msg.str());
  }

  BackendLib backend_lib(backend_lib_path);

  std::string loader_fn_name = "create_dynamic_loader_" + backend;
  std::stringstream err;
  if (!backend_lib.setLoader(loader_fn_name, err)) {
    throw std::runtime_error(err.str());
  }
  return backend_lib;
}

/*
BackendRegistry is a singleton class that manages the loading and unloading of
backend libraries. It maintains a map of backend names to backend libraries'
handle. In order to save resources, it loads libraries permanently, until the
handle get destroyed.
*/
struct BackendRegistry {
 public:
  BackendRegistry() = default;
  ~BackendRegistry() = default;

  DynamicLoaderInterface getBackend(const std::string& backend) {
    auto it = libs_.find(backend);
    if (it == libs_.end()) {
      auto res = libs_.emplace(backend, getBackendLib(backend));
      if (!res.second) {
        throw std::runtime_error("Failed to load backend library");
      }
      it = res.first;
    }
    return it->second.getLoader();
  }

  void eraseBackend(const std::string& backend) {
    if (auto it = libs_.find(backend); it != libs_.end()) {
      libs_.erase(it);
    }
  }

  static BackendRegistry& get() {
    static BackendRegistry instance;
    return instance;
  }

 private:
  std::unordered_map<std::string, BackendLib> libs_;
};

} // namespace

std::shared_ptr<TorchCommBackend> TorchCommFactory::create_backend(
    const std::string& backend,
    at::Device device,
    const std::string& name,
    const CommOptions& options) {
  std::shared_ptr<TorchCommBackend> impl;

  for (const auto& [key, _] : backends_) {
    TC_LOG(INFO) << "Backend " << key << " is registered";
  }

  if (auto it = backends_.find(backend); it != backends_.end()) {
    impl = it->second();
  } else {
    impl = create_generic_backend(backend);
  }

  if (impl) {
    impl->init(device, name, options);
  }

  return impl;
}

std::shared_ptr<TorchCommBackend> TorchCommFactory::create_generic_backend(
    const std::string& backend) {
  auto loader = BackendRegistry::get().getBackend(backend);

  // Create the backend instance
  TorchCommBackend* raw_backend = loader.new_comm();
  if (raw_backend == nullptr) {
    BackendRegistry::get().eraseBackend(backend);
    throw std::runtime_error("Failed to create backend instance");
  }

  // Create a shared_ptr with custom deleter that calls destroy_comm
  auto deleter = [loader](TorchCommBackend* ptr) {
    if (ptr) {
      loader.destroy_comm(ptr);
    }
  };
  std::shared_ptr<TorchCommBackend> backend_instance(raw_backend, deleter);
  return backend_instance;
}

void TorchCommFactory::register_backend(
    const std::string& backend,
    const std::function<std::shared_ptr<TorchCommBackend>()>& loader_fn) {
  std::lock_guard<std::mutex> guard(mutex_);
  backends_.emplace(backend, loader_fn);
}

// Allocator factory methods implementation
std::shared_ptr<c10::Allocator> TorchCommFactory::get_allocator(
    const std::string& backend) {
  std::lock_guard<std::mutex> guard(mutex_);

  auto it = allocator_factories_.find(backend);
  if (it != allocator_factories_.end()) {
    return it->second();
  }

  TORCH_CHECK(false, "No allocator factory registered for backend: ", backend);
}

void TorchCommFactory::register_allocator_factory(
    const std::string& backend,
    const std::function<std::shared_ptr<c10::Allocator>()>& factory) {
  std::lock_guard<std::mutex> guard(mutex_);
  allocator_factories_.emplace(backend, factory);
}

TorchCommFactory& TorchCommFactory::get() {
  static TorchCommFactory instance;
  return instance;
}
} // namespace torch::comms
