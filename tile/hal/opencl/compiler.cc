// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/opencl/compiler.h"

#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <boost/filesystem.hpp>

#include "base/util/callback_map.h"
#include "base/util/compat.h"
#include "base/util/env.h"
#include "base/util/file.h"
#include "base/util/logging.h"
#include "base/util/uuid.h"
#include "tile/hal/opencl/cl_opt.h"
#include "tile/hal/opencl/emitocl.h"
#include "tile/hal/opencl/library.h"
#include "tile/lang/semprinter.h"

namespace fs = boost::filesystem;

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {
namespace {

// Represents a build-in-flight
class Build {
 public:
  static boost::future<std::unique_ptr<hal::Library>> Start(context::Activity activity,
                                                            const std::shared_ptr<DeviceState>& device_state,
                                                            CLObj<cl_program> program,
                                                            const std::vector<lang::KernelInfo>& kernel_info,
                                                            proto::BuildInfo binfo,
                                                            std::vector<context::proto::ActivityID> kernel_ids);

  Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
        const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
        std::vector<context::proto::ActivityID> kernel_ids);

 private:
  static void OnBuildComplete(cl_program program, void* handle) noexcept;

  void OnError();

  context::Activity activity_;
  std::shared_ptr<DeviceState> device_state_;
  std::unique_ptr<Library> library_;
  boost::promise<std::unique_ptr<hal::Library>> prom_;
  proto::BuildInfo binfo_;

  static PendingCallbackMap<Build> pending_;
};

PendingCallbackMap<Build> Build::pending_;

boost::future<std::unique_ptr<hal::Library>> Build::Start(context::Activity activity,
                                                          const std::shared_ptr<DeviceState>& device_state,
                                                          CLObj<cl_program> program,
                                                          const std::vector<lang::KernelInfo>& kernel_info,
                                                          proto::BuildInfo binfo,
                                                          std::vector<context::proto::ActivityID> kernel_ids) {
  auto build = std::make_unique<Build>(std::move(activity), device_state, std::move(program), kernel_info,
                                       std::move(binfo), std::move(kernel_ids));
  auto result = build->prom_.get_future();
  cl_device_id device_id = device_state->did();
  cl_program prog = build->library_->program().get();
  auto handle = Build::pending_.Acquire(std::move(build));
  Err err = ocl::BuildProgram(prog, 1, &device_id, "-cl-fast-relaxed-math -cl-mad-enable -cl-unsafe-math-optimizations",
                              &OnBuildComplete, handle);
  if (err) {
    LOG(WARNING) << "Failed to build program: " << err;
    OnBuildComplete(prog, handle);
  }

  return result;
}

Build::Build(context::Activity activity, const std::shared_ptr<DeviceState>& device_state, CLObj<cl_program> program,
             const std::vector<lang::KernelInfo>& kernel_info, proto::BuildInfo binfo,
             std::vector<context::proto::ActivityID> kernel_ids)
    : activity_{std::move(activity)},
      device_state_{device_state},
      library_{std::make_unique<Library>(device_state, std::move(program), kernel_info, std::move(kernel_ids))},
      binfo_{std::move(binfo)} {}

void Build::OnBuildComplete(cl_program program, void* handle) noexcept {
  auto build = Build::pending_.Release(handle);
  if (!build) {
    // no-op, this handle has already been processed.
    return;
  }

  try {
    cl_build_status status;
    Err::Check(ocl::GetProgramBuildInfo(build->library_->program().get(), build->device_state_->did(),
                                        CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, nullptr),
               "Unable to construct program build status");
    if (status == CL_BUILD_SUCCESS) {
      build->prom_.set_value(std::move(build->library_));
    } else {
      LOG(WARNING) << "Failed to build program";
      build->binfo_.set_cl_build_status(status);
      build->OnError();
    }
    build->activity_.AddMetadata(build->binfo_);
  } catch (...) {
    build->prom_.set_exception(boost::current_exception());
  }
}

std::string WithLineNumbers(const std::string& src) {
  std::stringstream ss_in(src);
  std::stringstream ss_out;
  size_t line_num = 1;
  std::string line;
  while (std::getline(ss_in, line, '\n')) {
    ss_out << std::setw(5) << line_num++ << ": " << line << "\n";
  }
  return ss_out.str();
}

void Build::OnError() {
  size_t len = 0;
  Err bi_err =
      ocl::GetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
  if (bi_err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to retrieve build log size: " << bi_err;
  } else {
    std::string buffer(len, '\0');
    bi_err = ocl::GetProgramBuildInfo(library_->program().get(), device_state_->did(), CL_PROGRAM_BUILD_LOG, len,
                                      const_cast<char*>(buffer.c_str()), nullptr);
    if (bi_err) {
      LOG(ERROR) << "Failed to retrieve build log: " << bi_err;
    } else {
      LOG(WARNING) << "Failed build log: " << buffer;
      LOG(WARNING) << "Code was: \n" << WithLineNumbers(binfo_.src());
      binfo_.set_log(buffer);
    }
  }
  throw std::runtime_error{"Unable to compile Tile program"};
}

}  // namespace

Compiler::Compiler(const std::shared_ptr<DeviceState>& device_state) : device_state_{device_state} {}

std::string k_subgroup_microkernels =  // NOLINT
    R"***(

#pragma OPENCL EXTENSION cl_intel_subgroups : enable

#define vector_load(x) as_float(intel_sub_group_block_read((const global int*) (&(x))))
#define vector_store(x, v) intel_sub_group_block_write((const global int*) (&(x)), as_uint(v))

)***";                                 // NOLINT

boost::future<std::unique_ptr<hal::Library>> Compiler::Build(const context::Context& ctx,
                                                             const std::vector<lang::KernelInfo>& kernel_info,
                                                             const hal::proto::HardwareSettings& settings) {
  std::vector<context::proto::ActivityID> kernel_ids;
  std::ostringstream code;

  if (!kernel_info.size()) {
    return boost::make_ready_future(std::unique_ptr<hal::Library>{
        std::make_unique<Library>(device_state_, nullptr, kernel_info, std::vector<context::proto::ActivityID>{})});
  }

  context::Activity activity{ctx, "tile::hal::opencl::Build"};

  bool cl_khr_fp16 = device_state_->HasDeviceExtension("cl_khr_fp16");
  if (cl_khr_fp16) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";
  }

  bool cl_khr_fp64 = device_state_->HasDeviceExtension("cl_khr_fp64");
  if (cl_khr_fp64) {
    code << "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
  }

  bool cl_intel_subgroups = device_state_->HasDeviceExtension("cl_intel_subgroups");
  if (cl_intel_subgroups) {
    code << k_subgroup_microkernels;
  }

  auto env_cache = env::Get("PLAIDML_OPENCL_CACHE");
  fs::path cache_dir;
  if (env_cache.length()) {
    VLOG(1) << "Using OpenCL cache directory: " << env_cache;
    cache_dir = env_cache;
  }
  std::set<std::string> knames;

  code << "typedef struct {\n";
  code << "  float d;\n";
  code << "} custom;\n";
  code << "#define CUSTOM_MAX as_custom(FLT_MAX)\n";
  code << "#define CUSTOM_MIN as_custom(FLT_MIN)\n";
  code << "__local custom __OVERLOADABLE__ as_custom(float a, int b) {\n";
  code << "  custom c;\n";
  code << "  vstore_half(a, 0, (half*)&c.d);\n";
  code << "  return c;\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ as_custom(float a) {\n";
  code << "  return as_custom(a, 32);\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ as_custom(int a) {\n";
  code << "  return as_custom((float)a);\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ as_custom(custom a) {\n";
  code << "  return a;\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ as_custom(custom a, int b) {\n";
  code << "  return a;\n";
  code << "}\n";
  code << "__local float __OVERLOADABLE__ as_float(custom a) {\n";
  code << "  return vload_half(0, (half*)&a.d);\n";
  code << "}\n";
  code << "__local uint __OVERLOADABLE__ as_uint(custom a) {\n";
  code << "  return as_uint(as_float(a));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ native_log(custom a) {\n";
  code << "  return as_custom(native_log(as_float(a)));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ native_exp(custom a) {\n";
  code << "  return as_custom(native_exp(as_float(a)));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ native_sqrt(custom a) {\n";
  code << "  return as_custom(native_sqrt(as_float(a)));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ add(custom a, custom b) {\n";
  code << "  return as_custom(as_float(a) + as_float(b));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ sub(custom a, custom b) {\n";
  code << "  return as_custom(as_float(a) - as_float(b));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ mul(custom a, custom b) {\n";
  code << "  return as_custom(as_float(a) * as_float(b));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ div(custom a, custom b) {\n";
  code << "  return as_custom(as_float(a) / as_float(b));\n";
  code << "}\n";
  code << "__local int __OVERLOADABLE__ le(custom a, custom b) {\n";
  code << "  return as_float(a) < as_float(b);\n";
  code << "}\n";
  code << "__local int __OVERLOADABLE__ ge(custom a, custom b) {\n";
  code << "  return as_float(a) > as_float(b);\n";
  code << "}\n";
  code << "__local int __OVERLOADABLE__ eq(custom a, custom b) {\n";
  code << "  return as_float(a) == as_float(b);\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ neg(custom a) {\n";
  code << "  return as_custom(-as_float(a));\n";
  code << "}\n";
  code << "__local custom __OVERLOADABLE__ select(custom a, custom b, custom c) {\n";
  code << "  return as_custom(select(as_float(a), as_float(b), (int)as_float(c)));\n";
  code << "}\n";

  for (const auto& ki : kernel_info) {
    context::Activity kbuild{activity.ctx(), "tile::hal::opencl::BuildKernel"};

    proto::KernelInfo kinfo;
    kinfo.set_kname(ki.kname);

    if (ki.ktype == lang::KernelType::kZero) {
      kinfo.set_src("// Builtin zero kernel");
    } else if (!knames.count(ki.kfunc->name)) {
      knames.insert(ki.kfunc->name);
      OptimizeKernel(ki, cl_khr_fp16, settings);

      Emit ocl{cl_khr_fp16, cl_khr_fp64};
      ocl.Visit(*ki.kfunc);
      std::string src = ki.comments + ocl.str();

      if (is_directory(cache_dir)) {
        fs::path src_path = (cache_dir / ki.kname).replace_extension("cl");
        if (fs::is_regular_file(src_path)) {
          VLOG(1) << "Reading OpenCL code from cache: " << src_path;
          src = ReadFile(src_path);
        } else {
          VLOG(1) << "Writing OpenCL code to cache: " << src_path;
          WriteFile(src_path, src);
        }
      } else {
        if (VLOG_IS_ON(4)) {
          sem::Print emit_debug(*ki.kfunc);
          VLOG(4) << "Generic debug kernel:";
          VLOG(4) << ki.comments;
          VLOG(4) << emit_debug.str();
        }
      }

      code << src;
      code << "\n\n";

      kinfo.set_src(src);
    } else {
      kinfo.set_src("// Duplicate");
    }

    *(kinfo.mutable_kinfo()) = ki.info;
    kbuild.AddMetadata(kinfo);

    kernel_ids.emplace_back(kbuild.ctx().activity_id());
  }

  proto::BuildInfo binfo;
  *binfo.mutable_device_id() = device_state_->id();
  binfo.set_src(code.str());
  const char* src = binfo.src().c_str();
  Err err;

  VLOG(4) << "Compiling OpenCL:\n" << WithLineNumbers(binfo.src());
  CLObj<cl_program> program = ocl::CreateProgramWithSource(device_state_->cl_ctx().get(), 1, &src, nullptr, err.ptr());
  if (!program) {
    throw std::runtime_error(std::string("creating an OpenCL program object: ") + err.str());
  }

  return Build::Start(std::move(activity), device_state_, std::move(program), kernel_info, std::move(binfo),
                      std::move(kernel_ids));
}

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
