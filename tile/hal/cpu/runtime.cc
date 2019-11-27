// Copyright 2017-2018 Intel Corporation.

#include "tile/hal/cpu/runtime.h"

#include <llvm/Support/DynamicLibrary.h>

#include <half.hpp>
#include <math.h>

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
typedef struct {
  float d;
} custom;
#define MARGIN 1.0f
custom as_custom(float a, int b) {
  custom c;
  c.d = a * MARGIN;
  VLOG(2) << "as_custom, " << a << ", " << b << ", " << c.d;
  return c;
}
custom as_custom_long(long a, int b) {
  return as_custom((float) a, b);
}
custom as_custom_int(int a, int b) {
  return as_custom((float) a, b);
}
custom as_custom_double(double a, int b) {
  return as_custom((float) a, b);
}
custom as_custom_custom(custom a, int b) {
  return a;
}
float as_float(custom a, int b) {
  float r = a.d / MARGIN;
  VLOG(2) << "as_float, " << a.d << ", " << b << ", " << r;
  return r;
}
float as_float_const(double a, int b) {
  return (float) a;
}
float as_float_bool(bool a, int b) {
  return (float) a;
}
float as_float_float(float a, int b) {
  return a;
}
custom add(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  float r = ta + tb;
  VLOG(1) << "add, " << ta << ", " << tb << ", " << r;
  return as_custom(r, 32);
}
custom sub(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  float r = ta - tb;
  VLOG(1) << "sub, " << ta << ", " << tb << ", " << r;
  return as_custom(r, 32);
}
custom mul(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  float r = ta * tb;
  VLOG(1) << "mul, " << ta << ", " << tb << ", " << r;
  return as_custom(r, 32);
}
custom mul_const(custom a, float b) {
  float ta = as_float(a, 32);
  float tb = b;
  float r = ta * tb;
  VLOG(1) << "mul_const, " << ta << ", " << tb << ", " << r;
  return as_custom(r, 32);
}
custom div(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  float r = ta / tb;
  VLOG(1) << "div, " << ta << ", " << tb << ", " << r;
  return as_custom(r, 32);
}
custom neg(custom a) {
  float ta = as_float(a, 32);
  float r = - ta;
  VLOG(1) << "neg, " << ta << ", " << 0 << ", " << r;
  return as_custom(r, 32);
}
custom _exp(custom a) {
  float ta = as_float(a, 32);
  float r = exp(ta);
  VLOG(1) << "exp, " << ta << ", " << 0 << ", " << r;
  return as_custom(r, 32);
}
custom _sqrt(custom a) {
  float ta = as_float(a, 32);
  float r = sqrt(ta);
  VLOG(1) << "sqrt, " << ta << ", " << 0 << ", " << r;
  return as_custom(r, 32);
}
custom _round(custom a) {
  float ta = as_float(a, 32);
  float r = round(ta);
  VLOG(1) << "round, " << ta << ", " << 0 << ", " << r;
  return as_custom(r, 32);
}
bool lt(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  bool r = (ta < tb);
  VLOG(1) << "lt, " << ta << ", " << tb << ", " << r;
  return r;
}
bool eq(custom a, custom b) {
  float ta = as_float(a, 32);
  float tb = as_float(b, 32);
  bool r = (ta == tb);
  VLOG(1) << "eq, " << ta << ", " << tb << ", " << r;
  return r;
}
custom select_bool_fp32_custom(bool a, float b, custom c) {
  return (a) ? as_custom(b, 32) : c;
}
custom select_bool_i32_custom(bool a, int b, custom c) {
  return (a) ? as_custom((float)b, 32) : c;
}
custom select_bool_custom_custom(bool a, custom b, custom c) {
  return (a) ? b : c;
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
      {"as_custom_fp32_i32", symInfo(rt::as_custom)}, {"as_custom_i64_i32", symInfo(rt::as_custom_long)},
      {"as_custom_i32_i32", symInfo(rt::as_custom_int)}, {"as_custom_fp64_i32", symInfo(rt::as_custom_double)},
      {"as_custom_custom_i32", symInfo(rt::as_custom_custom)},
      {"as_float_fp32_i32", symInfo(rt::as_float_float)}, {"as_float_custom_i32", symInfo(rt::as_float)},
      {"as_float_i64_i32", symInfo(rt::as_float_const)}, {"as_float_bool_i32", symInfo(rt::as_float_bool)},
      {"add_custom_custom", symInfo(rt::add)}, {"mul_custom_custom", symInfo(rt::mul)},
      {"mul_custom_fp32", symInfo(rt::mul_const)},
      {"sub_custom_custom", symInfo(rt::sub)}, {"div_custom_custom", symInfo(rt::div)},
      {"lt_custom_custom", symInfo(rt::lt)}, {"eq_custom_custom", symInfo(rt::eq)},
      {"select_bool_fp32_custom", symInfo(rt::select_bool_fp32_custom)},
      {"select_bool_i32_custom", symInfo(rt::select_bool_i32_custom)},
      {"select_bool_custom_custom", symInfo(rt::select_bool_custom_custom)},
      {"neg_custom", symInfo(rt::neg)}, {"exp_custom", symInfo(rt::_exp)},
      {"sqrt_custom", symInfo(rt::_sqrt)}, {"round_custom", symInfo(rt::_round)},
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
