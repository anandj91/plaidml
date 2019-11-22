// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/runtime.h"

#include <llvm/Support/DynamicLibrary.h>

#include <half.hpp>

namespace vertexai {
namespace tile {
namespace hal {
namespace cpu {

namespace rt {
// Implementations of support functions the tile backend will link against,
// that we won't be able to resolve from system libraries.
void barrier() {}
float h2f(half_float::half n) { return n; }
half_float::half f2h(float n) { return half_float::half_cast<half_float::half>(n); }
inline void print(std::string op, float a, float b, float r) {
  std::stringstream stream;
  stream << "VALS " << a << op << b << " = " << r << std::endl;
  std::cout << stream.str();
}
float _add(float a, float b) {
  float r = a + b;
  print(" + ", a, b, r);
  return r;
}
float _sub(float a, float b) {
  float r = a - b;
  print(" - ", a, b, r);
  return r;
}
float _mul(float a, float b) {
  float r = a * b;
  print(" * ", a, b, r);
  return r;
}
float _div(float a, float b) {
  float r = a / b;
  print(" / ", a, b, r);
  return r;
}
float _neg(float a) {
  float r = -a;
  print(" - ", 0, a, r);
  return r;
}
}  // namespace rt

template <typename T>
llvm::JITEvaluatedSymbol symInfo(T ptr) {
  auto flags = llvm::JITSymbolFlags::None;
  auto addr = reinterpret_cast<uintptr_t>(ptr);
  return llvm::JITEvaluatedSymbol(addr, flags);
}

llvm::JITSymbol Runtime::findSymbol(const std::string& name) {
  static std::map<std::string, llvm::JITEvaluatedSymbol> symbols{
      {"Barrier", symInfo(rt::barrier)},   {"__gnu_h2f_ieee", symInfo(rt::h2f)}, {"__gnu_f2h_ieee", symInfo(rt::f2h)},
      {"___truncsfhf2", symInfo(rt::f2h)}, {"___extendhfsf2", symInfo(rt::h2f)},
      {"_add", symInfo(rt::_add)}, {"_mul", symInfo(rt::_mul)}, {"_sub", symInfo(rt::_sub)}, {"_div", symInfo(rt::_div)},
      {"_neg", symInfo(rt::_neg)},
  };
  auto loc = symbols.find(name);
  if (loc != symbols.end()) {
    return loc->second;
  }
  VLOG(4) << "cpu runtime resolving external symbol \"" << name << "\"";
  auto ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name);
  // If we failed to resolve the symbol, and its first character is an underscore, try again without
  // the underscore, because the code may have been generated for a system whose loader expects every
  // symbol to have an underscore prefix, but the DynamicLibrary module expects not to have a prefix.
  if (!ptr && name[0] == '_' && name.size() > 1) {
    ptr = llvm::sys::DynamicLibrary::SearchForAddressOfSymbol(name.substr(1));
  }
  if (ptr) {
    auto info = symInfo(ptr);
    symbols.emplace(name, info);
    return info;
  }
  std::string msg("cpu runtime failed to resolve external symbol reference: \"" + name + "\"");
  VLOG(1) << msg;
  throw(msg);
}

llvm::JITSymbol Runtime::findSymbolInLogicalDylib(const std::string& name) { return llvm::JITSymbol(nullptr); }

}  // namespace cpu
}  // namespace hal
}  // namespace tile
}  // namespace vertexai
