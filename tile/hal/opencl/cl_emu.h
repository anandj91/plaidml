#pragma once

#include <vector>

#include "tile/base/hal.h"
#include "tile/lang/generate.h"

namespace vertexai {
namespace tile {
namespace hal {
namespace opencl {

std::string EmulateType(const lang::KernelInfo& ki, std::vector<DataType> types, bool cl_khr_fp16, const hal::proto::HardwareSettings& settings);

}  // namespace opencl
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
