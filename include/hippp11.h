
// =================================================================================================
// This file is part of the CLCudaAPI project. The project is licensed under Apache Version 2.0. The
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements a bunch of C++11 classes that act as wrappers around CUDA objects and API
// calls. The main benefits are increased abstraction, automatic memory management, and portability.
// Portability here means that a similar header exists for OpenCL with the same classes and
// interfaces. In other words, moving from the CUDA API to the OpenCL API becomes a one-line change.
//
// This is version 9.0 of CLCudaAPI.
//
// =================================================================================================
//
// Copyright 2015 SURFsara
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//  http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// =================================================================================================

#ifndef CLCUDAAPI_CUPP11_H_
#define CLCUDAAPI_CUPP11_H_

// C++
#include <algorithm> // std::copy
#include <string>    // std::string
#include <vector>    // std::vector
#include <memory>    // std::shared_ptr

// HIP
#include <hip/hip_runtime.h>    // HIP driver API
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>   // HIP runtime compilation API

// Exception classes
#include "cxpp11_common.hpp"

namespace CLCudaAPI {
// =================================================================================================

// Max-length of strings
constexpr auto kStringLength = 256;

// =================================================================================================

// Represents a runtime error returned by a CUDA driver API function
class CLCudaAPIError : public ErrorCode<DeviceError, hipError_t> {
public:
  explicit CLCudaAPIError(hipError_t status, const std::string &where):
      ErrorCode(status, where, "CUDA error: " + where + ": " +
                               GetErrorName(status) + " --> " + GetErrorString(status)) {
  }

  static void Check(const hipError_t status, const std::string &where) {
    if (status != hipSuccess) {
      throw CLCudaAPIError(status, where);
    }
  }

  static void CheckDtor(const hipError_t status, const std::string &where) {
    if (status != hipSuccess) {
      fprintf(stderr, "CLCudaAPI: %s (ignoring)\n", CLCudaAPIError(status, where).what());
    }
  }

private:
  std::string GetErrorName(hipError_t status) const {
    //const char* status_code;
    //hipDrvGetErrorName(status, &status_code);
    const char* status_code = hipGetErrorName(status);
    return std::string(status_code);
  }
  std::string GetErrorString(hipError_t status) const {
    //const char* status_string;
    //hipDrvGetErrorString(status, &status_string);
    const char* status_string = hipGetErrorString(status);
    return std::string(status_string);
  }
};

// Represents a runtime error returned by a CUDA runtime compilation API function
class CLCudaAPINVRTCError : public ErrorCode<DeviceError, hiprtcResult> {
public:
  explicit CLCudaAPINVRTCError(hiprtcResult status, const std::string &where):
      ErrorCode(status, where, "HIP RTC error: " + where + ": " + GetErrorString(status)) {
  }

  static void Check(const hiprtcResult status, const std::string &where) {
    if (status != HIPRTC_SUCCESS) {
      throw CLCudaAPINVRTCError(status, where);
    }
  }

  static void CheckDtor(const hiprtcResult status, const std::string &where) {
    if (status != HIPRTC_SUCCESS) {
      fprintf(stderr, "CLCudaAPI: %s (ignoring)\n", CLCudaAPINVRTCError(status, where).what());
    }
  }

private:
  std::string GetErrorString(hiprtcResult status) const {
    const char* status_string = hiprtcGetErrorString(status);
    return std::string(status_string);
  }
};

// Exception returned when building a program
using CLCudaAPIBuildError = CLCudaAPINVRTCError;

// =================================================================================================

// Error occurred in CUDA driver or runtime compilation API
#define CheckError(call) CLCudaAPIError::Check(call, CLCudaAPIError::TrimCallString(#call))
#define CheckErrorNVRTC(call) CLCudaAPINVRTCError::Check(call, CLCudaAPINVRTCError::TrimCallString(#call))

// Error occurred in CUDA driver or runtime compilation API (no-exception version for destructors)
#define CheckErrorDtor(call) CLCudaAPIError::CheckDtor(call, CLCudaAPIError::TrimCallString(#call))
#define CheckErrorDtorNVRTC(call) CLCudaAPINVRTCError::CheckDtor(call, CLCudaAPINVRTCError::TrimCallString(#call))

// =================================================================================================

// C++11 version of two 'hipEvent_t' pointers
class Event {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Regular constructor with memory management
  explicit Event():
      start_(new hipEvent_t, [](hipEvent_t* e) { CheckErrorDtor(hipEventDestroy(*e)); delete e; }),
      end_(new hipEvent_t, [](hipEvent_t* e) { CheckErrorDtor(hipEventDestroy(*e)); delete e; }) {
    CheckError(hipEventCreateWithFlags(start_.get(), hipEventDefault));
    CheckError(hipEventCreateWithFlags(end_.get(), hipEventDefault));
  }

  // Waits for completion of this event (not implemented for CUDA)
  void WaitForCompletion() const { }

  // Retrieves the elapsed time of the last recorded event
  float GetElapsedTime() const {
    auto result = 0.0f;
    hipEventElapsedTime(&result, *start_, *end_);
    return result;
  }

  // Accessors to the private data-members
  const hipEvent_t& start() const { return *start_; }
  const hipEvent_t& end() const { return *end_; }
  Event* pointer() { return this; }
 private:
  std::shared_ptr<hipEvent_t> start_;
  std::shared_ptr<hipEvent_t> end_;
};

// Pointer to a CUDA event
using EventPointer = Event*;

// =================================================================================================

// Raw platform ID type
using RawPlatformID = size_t;

// The CUDA platform: initializes the CUDA driver API
class Platform {
 public:

  // Initializes the platform. Note that the platform ID variable is not actually used for CUDA.
  explicit Platform(const size_t platform_id) : platform_id_(0) {
    if (platform_id != 0) { throw LogicError("HIP back-end requires a platform ID of 0"); }
    CheckError(hipInit(0));
  }

  // Methods to retrieve platform information
  std::string Name() const { return "HIP"; }
  std::string Vendor() const { return "AMD"; }
  std::string Version() const {
    auto result = 0;
    CheckError(hipDriverGetVersion(&result));
    return "HIP driver "+std::to_string(result);
  }

  // Returns the number of devices on this platform
  size_t NumDevices() const {
    auto result = 0;
    CheckError(hipGetDeviceCount(&result));
    return static_cast<size_t>(result);
  }

  // Accessor to the raw ID (which doesn't exist in the CUDA back-end, this is always just 0)
  const RawPlatformID& operator()() const { return platform_id_; }
private:
  const size_t platform_id_;
};

// Retrieves a vector with all platforms. Note that there is just one platform in CUDA.
inline std::vector<Platform> GetAllPlatforms() {
  auto all_platforms = std::vector<Platform>{ Platform(size_t{0}) };
  return all_platforms;
}

// =================================================================================================

// Raw device ID type
using RawDeviceID = hipDevice_t;

// C++11 version of 'hipDevice_t'
class Device {
 public:

  // Constructor based on the regular CUDA data-type
  explicit Device(const hipDevice_t device): device_(device) { }

  // Initialization
  explicit Device(const Platform &platform, const size_t device_id) {
    auto num_devices = platform.NumDevices();
    if (num_devices == 0) {
      throw RuntimeError("Device: no devices found");
    }
    if (device_id >= num_devices) {
      throw RuntimeError("Device: invalid device ID "+std::to_string(device_id));
    }

    CheckError(hipDeviceGet(&device_, device_id));
  }

  // Methods to retrieve device information
  RawPlatformID PlatformID() const { return 0; }
  std::string Version() const {
    auto result = 0;
    CheckError(hipDriverGetVersion(&result));
    return "HIP driver "+std::to_string(result);
  }
  size_t VersionNumber() const {
    auto result = 0;
    CheckError(hipDriverGetVersion(&result));
    return static_cast<size_t>(result);
  }
  std::string Vendor() const { return "AMD"; }
  std::string Name() const {
    auto result = std::string{};
    result.resize(kStringLength);
    CheckError(hipDeviceGetName(&result[0], result.size(), device_));
    return result;
  }
  std::string Type() const { return "GPU"; }
  size_t MaxWorkGroupSize() const {return GetInfo(hipDeviceAttributeMaxThreadsPerBlock); }
  size_t MaxWorkItemDimensions() const { return size_t{3}; }
  std::vector<size_t> MaxWorkItemSizes() const {
    return std::vector<size_t>{GetInfo(hipDeviceAttributeMaxBlockDimX),
                               GetInfo(hipDeviceAttributeMaxBlockDimY),
                               GetInfo(hipDeviceAttributeMaxBlockDimZ)};
  }
  unsigned long LocalMemSize() const {
    return static_cast<unsigned long>(GetInfo(hipDeviceAttributeMaxSharedMemoryPerBlock));
  }

  std::string Capabilities() const {
    const auto major = GetInfo(hipDeviceAttributeComputeCapabilityMajor);
    const auto minor = GetInfo(hipDeviceAttributeComputeCapabilityMinor);
    return "SM"+std::to_string(major)+"."+std::to_string(minor);
  }
  bool HasExtension(const std::string &extension) const { return false; }
  bool SupportsFP64() const { return true; }
  bool SupportsFP16() const {
    const auto major = GetInfo(hipDeviceAttributeComputeCapabilityMajor);
    const auto minor = GetInfo(hipDeviceAttributeComputeCapabilityMinor);
    if (major > 5) { return true; } // SM 6.x, 7.x and higher
    if (major == 5 && minor == 3) { return true; } // SM 5.3
    return false;
  }

  size_t CoreClock() const { return 1e-3*GetInfo(hipDeviceAttributeClockRate); }
  size_t ComputeUnits() const { return GetInfo(hipDeviceAttributeMultiprocessorCount); }
  unsigned long MemorySize() const {
    auto result = size_t{0};
    CheckError(hipDeviceTotalMem(&result, device_));
    return static_cast<unsigned long>(result);
  }
  unsigned long MaxAllocSize() const { return MemorySize(); }
  size_t MemoryClock() const { return 1e-3*GetInfo(hipDeviceAttributeMemoryClockRate); }
  size_t MemoryBusWidth() const { return GetInfo(hipDeviceAttributeMemoryBusWidth); }

  // Configuration-validity checks
  bool IsLocalMemoryValid(const size_t local_mem_usage) const {
    return (local_mem_usage <= LocalMemSize());
  }
  bool IsThreadConfigValid(const std::vector<size_t> &local) const {
    auto local_size = size_t{1};
    for (const auto &item: local) { local_size *= item; }
    for (auto i=size_t{0}; i<local.size(); ++i) {
      if (local[i] > MaxWorkItemSizes()[i]) { return false; }
    }
    if (local_size > MaxWorkGroupSize()) { return false; }
    if (local.size() > MaxWorkItemDimensions()) { return false; }
    return true;
  }

  // Query for a specific type of device or brand
  bool IsCPU() const { return false; }
  bool IsGPU() const { return true; }
  bool IsAMD() const { return true; }
  bool IsNVIDIA() const { return false; }
  bool IsIntel() const { return false; }
  bool IsARM() const { return false; }

  // Platform specific extensions
  std::string AMDBoardName() const { return ""; }
  std::string NVIDIAComputeCapability() const { return Capabilities(); }

  // Accessor to the private data-member
  const RawDeviceID& operator()() const { return device_; }
 private:
  hipDevice_t device_;

  // Private helper function
  size_t GetInfo(const hipDeviceAttribute_t info) const {
    auto result = 0;
    CheckError(hipDeviceGetAttribute(&result, info, device_));
    return static_cast<size_t>(result);
  }
};

// =================================================================================================

// Raw context type
using RawContext = hipCtx_t;

// C++11 version of 'hipCtx_t'
class Context {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Context(const hipCtx_t context):
      context_(new hipCtx_t) {
    *context_ = context;
  }

  // Regular constructor with memory management
  explicit Context(const Device &device):
      context_(new hipCtx_t, [](hipCtx_t* c) {
        if (*c) { CheckErrorDtor(hipCtxDestroy(*c)); }
        delete c;
      }) {
    CheckError(hipCtxCreate(context_.get(), 0, device()));
  }

  // Accessor to the private data-member
  const RawContext& operator()() const { return *context_; }
  RawContext* pointer() const { return &(*context_); }
 private:
  std::shared_ptr<hipCtx_t> context_;
};

// Pointer to a raw CUDA context
using ContextPointer = hipCtx_t*;

// =================================================================================================

// C++11 version of 'hiprtcProgram'. Additionally holds the program's source code.
class Program {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Source-based constructor with memory management
  explicit Program(const Context &, std::string source):
      program_(new hiprtcProgram, [](hiprtcProgram* p) {
          if (*p) { CheckErrorDtorNVRTC(hiprtcDestroyProgram(p)); }
          delete p;
      }),
      source_(std::move(source)),
      from_binary_(false) {
    const auto source_ptr = &source_[0];
    CheckErrorNVRTC(hiprtcCreateProgram(program_.get(), source_ptr, nullptr, 0, nullptr, nullptr));
  }

  // PTX-based constructor
  explicit Program(const Device &device, const Context &context, const std::string &binary):
      program_(nullptr), // not used
      source_(binary),
      from_binary_(true) {
  }

  // Compiles the device program and checks whether or not there are any warnings/errors
  void Build(const Device &, std::vector<std::string> &options) {
    if (from_binary_) { return; }
    auto raw_options = std::vector<const char*>();
    for (const auto &option: options) {
      raw_options.push_back(option.c_str());
    }
    auto status = hiprtcCompileProgram(*program_, raw_options.size(), raw_options.data());
    CLCudaAPINVRTCError::Check(status, "hiprtcCompileProgram");
  }

  // Confirms whether a certain status code is an actual compilation error or warning
  bool StatusIsCompilationWarningOrError(const hiprtcResult status) const {
    return (status == HIPRTC_ERROR_INVALID_INPUT);
  }

  // Retrieves the warning/error message from the compiler (if any)
  std::string GetBuildInfo(const Device &) const {
    if (from_binary_) { return std::string{}; }
    auto bytes = size_t{0};
    CheckErrorNVRTC(hiprtcGetProgramLogSize(*program_, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckErrorNVRTC(hiprtcGetProgramLog(*program_, &result[0]));
    return result;
  }

  // Retrieves an intermediate representation of the compiled program (i.e. PTX)
  std::string GetIR() const {
    if (from_binary_) { return source_; } // holds the PTX
    auto bytes = size_t{0};
    CheckErrorNVRTC(hiprtcGetCodeSize(*program_, &bytes));
    auto result = std::string{};
    result.resize(bytes);
    CheckErrorNVRTC(hiprtcGetCode(*program_, &result[0]));
    return result;
  }

  // Accessor to the private data-member
  const hiprtcProgram& operator()() const { return *program_; }
 private:
  std::shared_ptr<hiprtcProgram> program_;
  const std::string source_;
  const bool from_binary_;
};

// =================================================================================================

// Raw command-queue type
using RawCommandQueue = hipStream_t;

// C++11 version of 'hipStream_t'
class Queue {
 public:
  // Note that there is no constructor based on the regular CUDA data-type because of extra state

  // Regular constructor with memory management
  explicit Queue(const Context &context, const Device &device):
      queue_(new hipStream_t, [](hipStream_t* s) {
        if (*s) { CheckErrorDtor(hipStreamDestroy(*s)); }
        delete s;
      }),
      context_(context),
      device_(device) {
    CheckError(hipStreamCreateWithFlags(queue_.get(), hipStreamNonBlocking));
  }

  // Synchronizes the queue and optionally also an event
  void Finish(Event &event) const {
    CheckError(hipEventSynchronize(event.end()));
    Finish();
  }
  void Finish() const {
    CheckError(hipStreamSynchronize(*queue_));
  }

  // Retrieves the corresponding context or device
  Context GetContext() const { return context_; }
  Device GetDevice() const { return device_; }

  // Accessor to the private data-member
  const RawCommandQueue& operator()() const { return *queue_; }
 private:
  std::shared_ptr<hipStream_t> queue_;
  const Context context_;
  const Device device_;
};

// =================================================================================================

// C++11 version of page-locked host memory
template <typename T>
class BufferHost {
 public:

  // Regular constructor with memory management
  explicit BufferHost(const Context &, const size_t size):
      buffer_(new void*, [](void** m) { CheckError(hipHostFree(*m)); delete m; }),
      size_(size) {
    CheckError(hipHostMalloc(buffer_.get(), size*sizeof(T)));
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    return size_*sizeof(T);
  }

  // Compatibility with std::vector
  size_t size() const { return size_; }
  T* begin() { return &static_cast<T*>(*buffer_)[0]; }
  T* end() { return &static_cast<T*>(*buffer_)[size_-1]; }
  T& operator[](const size_t i) { return static_cast<T*>(*buffer_)[i]; }
  T* data() { return static_cast<T*>(*buffer_); }
  const T* data() const { return static_cast<T*>(*buffer_); }

 private:
  std::shared_ptr<void*> buffer_;
  const size_t size_;
};

// =================================================================================================

// Enumeration of buffer access types
enum class BufferAccess { kReadOnly, kWriteOnly, kReadWrite, kNotOwned };

// C++11 version of 'hipDeviceptr_t'
template <typename T>
class Buffer {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Buffer(const hipDeviceptr_t buffer):
      buffer_(new hipDeviceptr_t),
      access_(BufferAccess::kNotOwned) {
    *buffer_ = buffer;
  }

  // Regular constructor with memory management. If this class does not own the buffer object, then
  // the memory will not be freed automatically afterwards.
  explicit Buffer(const Context &, const BufferAccess access, const size_t size):
      buffer_(new hipDeviceptr_t, [access](hipDeviceptr_t* m) {
        if (access != BufferAccess::kNotOwned) { CheckError(hipFree(*m)); }
        delete m;
      }),
      access_(access) {
    CheckError(hipMalloc(buffer_.get(), size*sizeof(T)));
  }

  // As above, but now with read/write access as a default
  explicit Buffer(const Context &context, const size_t size):
    Buffer<T>(context, BufferAccess::kReadWrite, size) {
  }

  // Constructs a new buffer based on an existing host-container
  template <typename Iterator>
  explicit Buffer(const Context &context, const Queue &queue, Iterator start, Iterator end):
    Buffer(context, BufferAccess::kReadWrite, static_cast<size_t>(end - start)) {
    auto size = static_cast<size_t>(end - start);
    auto pointer = &*start;
    CheckError(hipMemcpyHtoDAsync(*buffer_, pointer, size*sizeof(T), queue()));
    queue.Finish();
  }

  // Copies from device to host: reading the device buffer a-synchronously
  void ReadAsync(const Queue &queue, const size_t size, T* host, const size_t offset = 0) const {
    if (access_ == BufferAccess::kWriteOnly) {
      throw LogicError("Buffer: reading from a write-only buffer");
    }
    CheckError(hipMemcpyDtoHAsync(host, static_cast<T*>(*buffer_) + offset, size*sizeof(T), queue()));
  }
  void ReadAsync(const Queue &queue, const size_t size, std::vector<T> &host,
                 const size_t offset = 0) const {
    if (host.size() < size) {
      throw LogicError("Buffer: target host buffer is too small");
    }
    ReadAsync(queue, size, host.data(), offset);
  }
  void ReadAsync(const Queue &queue, const size_t size, BufferHost<T> &host,
                 const size_t offset = 0) const {
    if (host.size() < size) {
      throw LogicError("Buffer: target host buffer is too small");
    }
    ReadAsync(queue, size, host.data(), offset);
  }

  // Copies from device to host: reading the device buffer
  void Read(const Queue &queue, const size_t size, T* host, const size_t offset = 0) const {
    ReadAsync(queue, size, host, offset);
    queue.Finish();
  }
  void Read(const Queue &queue, const size_t size, std::vector<T> &host,
            const size_t offset = 0) const {
    Read(queue, size, host.data(), offset);
  }
  void Read(const Queue &queue, const size_t size, BufferHost<T> &host,
            const size_t offset = 0) const {
    Read(queue, size, host.data(), offset);
  }

  // Copies from host to device: writing the device buffer a-synchronously
  void WriteAsync(const Queue &queue, const size_t size, const T* host_cost, const size_t offset = 0) {
    if (access_ == BufferAccess::kReadOnly) {
      throw LogicError("Buffer: writing to a read-only buffer");
    }
    if (GetSize() < (offset+size)*sizeof(T)) {
      throw LogicError("Buffer: target device buffer is too small");
    }
    // hipMemcpyHtoDAsync does not change the host data, but is not defined with const pointer!
    T* host = const_cast<T*>(host_cost);
    CheckError(hipMemcpyHtoDAsync(static_cast<T*>(*buffer_) + offset, host, size*sizeof(T), queue()));
  }
  void WriteAsync(const Queue &queue, const size_t size, const std::vector<T> &host,
                  const size_t offset = 0) {
    WriteAsync(queue, size, host.data(), offset);
  }
  void WriteAsync(const Queue &queue, const size_t size, const BufferHost<T> &host,
                  const size_t offset = 0) {
    WriteAsync(queue, size, host.data(), offset);
  }

  // Copies from host to device: writing the device buffer 
  void Write(const Queue &queue, const size_t size, T* host, const size_t offset = 0) {
    WriteAsync(queue, size, host, offset);
    queue.Finish();
  }
  void Write(const Queue &queue, const size_t size, std::vector<T> &host,
             const size_t offset = 0) {
    Write(queue, size, host.data(), offset);
  }
  void Write(const Queue &queue, const size_t size, const BufferHost<T> &host,
             const size_t offset = 0) {
    Write(queue, size, host.data(), offset);
  }

  // Copies the contents of this buffer into another device buffer
  void CopyToAsync(const Queue &queue, const size_t size, const Buffer<T> &destination) const {
    CheckError(hipMemcpyDtoDAsync(destination(), *buffer_, size*sizeof(T), queue()));
  }
  void CopyTo(const Queue &queue, const size_t size, const Buffer<T> &destination) const {
    CopyToAsync(queue, size, destination);
    queue.Finish();
  }

  // Retrieves the actual allocated size in bytes
  size_t GetSize() const {
    auto result = size_t{0};
    CheckError(hipMemGetAddressRange(nullptr, &result, *buffer_));
    return result;
  }

  // Accessors to the private data-members
  hipDeviceptr_t operator()() const { return *buffer_; }
  hipDeviceptr_t& operator()() { return *buffer_; }
 private:
  std::shared_ptr<hipDeviceptr_t> buffer_;
  const BufferAccess access_;
};

// =================================================================================================

// C++11 version of 'hipFunction_t'
class Kernel {
 public:

  // Constructor based on the regular CUDA data-type: memory management is handled elsewhere
  explicit Kernel(const hipModule_t module, const hipFunction_t kernel):
    module_(module),
    kernel_(kernel) {
  }

  // Regular constructor with memory management
  explicit Kernel(const Program &program, const std::string &name) {
    CheckError(hipModuleLoadDataEx(&module_, program.GetIR().data(), 0, nullptr, nullptr));
    CheckError(hipModuleGetFunction(&kernel_, module_, name.c_str()));
  }

  // Sets a kernel argument at the indicated position. This stores both the value of the argument
  // (as raw bytes) and the index indicating where this value can be found.
  template <typename T>
  void SetArgument(const size_t index, const T &value) {
    if (index >= arguments_indices_.size()) { arguments_indices_.resize(index+1); }
    arguments_indices_[index] = arguments_data_.size();
    for (auto j=size_t(0); j<sizeof(T); ++j) {
      arguments_data_.push_back(reinterpret_cast<const char*>(&value)[j]);
    }
  }
  template <typename T>
  void SetArgument(const size_t index, Buffer<T> &value) {
    SetArgument(index, value());
  }

  // Sets all arguments in one go using parameter packs. Note that this resets all previously set
  // arguments using 'SetArgument' or 'SetArguments'.
  template <typename... Args>
  void SetArguments(Args&... args) {
    arguments_indices_.clear();
    arguments_data_.clear();
    SetArgumentsRecursive(0, args...);
  }

  // Retrieves the amount of local memory used per work-group for this kernel. Note that this the
  // shared memory in CUDA terminology.
  unsigned long LocalMemUsage(const Device &) const {
    auto result = 0;
    CheckError(hipFuncGetAttribute(&result, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, kernel_));
    return static_cast<unsigned long>(result);
  }

  // Retrieves the name of the kernel
  std::string GetFunctionName() const {
    return std::string{"unknown"}; // Not implemented for the CUDA backend
  }

  // Launches a kernel onto the specified queue
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event) {

    // Creates the grid (number of threadblocks) and sets the block sizes (threads per block)
    auto grid = std::vector<size_t>{1, 1, 1};
    auto block = std::vector<size_t>{1, 1, 1};
    if (global.size() != local.size()) { throw LogicError("invalid thread/workgroup dimensions"); }
    for (auto i=size_t{0}; i<local.size(); ++i) { grid[i] = global[i]/local[i]; }
    for (auto i=size_t{0}; i<local.size(); ++i) { block[i] = local[i]; }

    // Creates the array of pointers from the arrays of indices & data
    std::vector<void*> pointers;
    for (auto &index: arguments_indices_) {
      pointers.push_back(&arguments_data_[index]);
    }

    // Launches the kernel, its execution time is recorded by events
    CheckError(hipEventRecord(event->start(), queue()));
    CheckError(hipModuleLaunchKernel(kernel_, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                              0, queue(), pointers.data(), nullptr));
    CheckError(hipEventRecord(event->end(), queue()));
  }

  // As above, but with an event waiting list
  // TODO: Implement this function
  void Launch(const Queue &queue, const std::vector<size_t> &global,
              const std::vector<size_t> &local, EventPointer event,
              std::vector<Event>& waitForEvents) {
    if (local.size() == 0) {
      throw LogicError("Kernel: launching with a default workgroup size is not implemented for the CUDA back-end");
    }
    else if (waitForEvents.size() != 0) {
      throw LogicError("Kernel: launching with an event waiting list is not implemented for the CUDA back-end");
    }
    else {
     return Launch(queue, global, local, event);
    }
  }

  // Accessors to the private data-members
  const hipFunction_t& operator()() const { return kernel_; }
  hipFunction_t operator()() { return kernel_; }
 private:
  hipModule_t module_;
  hipFunction_t kernel_;
  std::vector<size_t> arguments_indices_; // Indices of the arguments
  std::vector<char> arguments_data_; // The arguments data as raw bytes

  // Internal implementation for the recursive SetArguments function.
  template <typename T>
  void SetArgumentsRecursive(const size_t index, T &first) {
    SetArgument(index, first);
  }
  template <typename T, typename... Args>
  void SetArgumentsRecursive(const size_t index, T &first, Args&... args) {
    SetArgument(index, first);
    SetArgumentsRecursive(index+1, args...);
  }
};

// =================================================================================================
} // namespace CLCudaAPI

// CLCUDAAPI_CUPP11_H_
#endif
