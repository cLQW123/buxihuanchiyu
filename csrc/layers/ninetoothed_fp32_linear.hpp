#pragma once
#include <infinicore/context/context.hpp>
#include <infinicore/ops/cast.hpp>
#include <infinicore/ops/distributed/allreduce.hpp>
#include <infinicore/tensor.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <dlfcn.h>
#include <stdexcept>
#include <string>

namespace infinilm::layers::ninetoothed_bridge {
struct TensorArgument {
    void *data;
    const uint64_t *shape;
    const int64_t *strides;
};
using MatmulLauncher = int (*)(void *, TensorArgument, TensorArgument,
                               TensorArgument, TensorArgument);

// The numerical kernel is generated from NineToothed. This adapter forms
// storage views and schedules FP32 communication before the output cast.
inline infinicore::Tensor linear_allreduce_fp32(
    const infinicore::Tensor &input, const infinicore::Tensor &weight,
    infinicclComm_t communicator) {
    using namespace infinicore;
    if (input->dtype() != DataType::BF16 || weight->dtype() != DataType::BF16 ||
        weight->ndim() != 2 || input->shape().back() != weight->size(1)) {
        throw std::invalid_argument("NineToothed FP32 linear requires matching BF16 matrix storage");
    }
    const char *directory = std::getenv("SGCUP_TP_MATMUL_KERNEL_DIR");
    if (!directory) throw std::runtime_error("SGCUP_TP_MATMUL_KERNEL_DIR is required");
    static thread_local MatmulLauncher launch = nullptr;
    if (!launch) {
        const std::string name = "sgcup_tp_matmul_f32_v158";
        void *library = dlopen((std::string(directory) + "/" + name + ".so").c_str(),
                               RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        launch = reinterpret_cast<MatmulLauncher>(dlsym(library, ("launch_" + name).c_str()));
        if (!launch) throw std::runtime_error("Missing NineToothed FP32 matmul launcher");
    }
    context::setDevice(input->device());
    const uint64_t k = weight->size(1), n = weight->size(0), rows = input->numel() / k;
    auto a = input->contiguous()->view({rows, k});
    auto b = weight->permute({1, 0});
    auto partial = Tensor::empty({rows, n}, DataType::F32, input->device());
    std::array<uint64_t, 2> ashape{rows, k}, bshape{k, n}, cshape{rows, n};
    std::array<int64_t, 2> astrides{static_cast<int64_t>(k), 1};
    std::array<int64_t, 2> bstrides{b->strides()[0], b->strides()[1]};
    std::array<int64_t, 2> cstrides{static_cast<int64_t>(n), 1};
    int status = launch(context::getStream(),
                        {a->data(), ashape.data(), astrides.data()},
                        {b->data(), bshape.data(), bstrides.data()},
                        {partial->data(), cshape.data(), cstrides.data()},
                        {nullptr, nullptr, nullptr});
    if (status) throw std::runtime_error("NineToothed FP32 matmul launch failed: " + std::to_string(status));
    op::distributed::allreduce_(partial, partial, INFINICCL_SUM, communicator);
    auto shape = input->shape();
    shape.back() = n;
    auto output = Tensor::empty(shape, input->dtype(), input->device());
    op::cast_(output, partial->view(shape));
    return output;
}
} // namespace infinilm::layers::ninetoothed_bridge
