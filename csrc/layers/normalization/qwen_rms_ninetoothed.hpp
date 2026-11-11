#pragma once
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/context/context.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <map>
#include <stdexcept>
#include <string>
namespace infinilm::layers::normalization {
struct QwenNTTensor { void *data; const uint64_t *shape; const int64_t *strides; };
using QwenNTLaunch = int (*)(void *, QwenNTTensor, QwenNTTensor, QwenNTTensor, QwenNTTensor);
inline infinicore::Tensor qwen_rms(const infinicore::Tensor &input, const std::shared_ptr<infinicore::nn::RMSNorm> &norm) {
 if (input->device().getType() != infinicore::Device::Type::HYGON) return norm->forward(input);
 const size_t width = input->shape().back();
 if (input->dtype() != infinicore::DataType::BF16 || (width != 128 && width != 5120)) throw std::runtime_error("Unsupported NineToothed Qwen RMSNorm dtype or width");
 const char *directory = std::getenv("SGCUP_QWEN_RMS_KERNEL_DIR");
 if (!directory) throw std::runtime_error("SGCUP_QWEN_RMS_KERNEL_DIR is required");
 static thread_local std::map<size_t,QwenNTLaunch> launches;
 auto found = launches.find(width);
 if (found == launches.end()) {
  const std::string name = "sgcup_qwen_rms_v140_" + std::to_string(width);
  void *library = dlopen((std::string(directory) + "/" + name + ".so").c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!library) throw std::runtime_error(dlerror());
  auto launch = reinterpret_cast<QwenNTLaunch>(dlsym(library, ("launch_" + name).c_str()));
  if (!launch) throw std::runtime_error("Missing NineToothed Qwen RMSNorm launcher");
  found = launches.emplace(width,launch).first;
 }
 auto x = input->contiguous()->view({input->numel()/width,width});
 auto y = infinicore::Tensor::empty(x->shape(),x->dtype(),x->device());
 auto weight = norm->weight()->contiguous();
 std::array<uint64_t,2> shape{input->numel()/width,width};
 std::array<int64_t,2> strides{static_cast<int64_t>(width),1};
 std::array<uint64_t,1> wshape{width}; std::array<int64_t,1> wstrides{1};
 double eps = norm->eps();
 infinicore::context::setDevice(input->device());
 int status = found->second(infinicore::context::getStream(),
   {x->data(),shape.data(),strides.data()}, {weight->data(),wshape.data(),wstrides.data()},
   {y->data(),shape.data(),strides.data()}, {&eps,nullptr,nullptr});
 if (status) throw std::runtime_error("NineToothed Qwen RMSNorm launch failed: " + std::to_string(status));
 return y->view(input->shape());
}
}
