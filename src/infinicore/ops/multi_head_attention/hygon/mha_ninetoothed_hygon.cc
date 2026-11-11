#if defined(ENABLE_HYGON_API) && defined(ENABLE_NINETOOTHED)
#include "infinicore/ops/mha.hpp"
#include "infinicore/ops/flash_attention.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace infinicore::op::mha_impl::ninetoothed_hygon {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, lengths;
    float scale;
    bool causal;
};

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           std::optional<Tensor> alibi, float scale, bool causal) {
    if (alibi.has_value()) {
        throw std::invalid_argument("NineToothed Hygon MHA does not support ALiBi");
    }
    if (q->ndim() != 4 || k->ndim() != 4 || v->ndim() != 4) {
        throw std::invalid_argument("NineToothed Hygon MHA requires BSHD tensors");
    }
    if (q->dtype() != DataType::BF16 || k->dtype() != q->dtype() || v->dtype() != q->dtype()) {
        throw std::invalid_argument("NineToothed Hygon MHA requires matching BF16 tensors");
    }
    if (q->size(0) != k->size(0) || k->shape() != v->shape() ||
        q->size(3) != k->size(3) || q->size(2) % k->size(2) != 0 ||
        out->shape() != q->shape() || k->size(1) > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Inconsistent NineToothed Hygon MHA tensor shapes");
    }
    auto host_lengths = Tensor::empty({q->size(0)}, DataType::I32, Device::cpu());
    std::fill_n(reinterpret_cast<int32_t *>(host_lengths->data()), q->size(0),
                static_cast<int32_t>(k->size(1)));
    auto lengths = host_lengths->to(q->device());
    context::syncStream();
    return new PlannedMeta{graph::GraphTensor(out), graph::GraphTensor(q),
                           graph::GraphTensor(k), graph::GraphTensor(v),
                           graph::GraphTensor(lengths), scale, causal};
}

void run(void *metadata) {
    auto *p = reinterpret_cast<PlannedMeta *>(metadata);
    flash_attention_(p->out->permute({0, 2, 1, 3}),
                     p->q->permute({0, 2, 1, 3}),
                     p->k->permute({0, 2, 1, 3}),
                     p->v->permute({0, 2, 1, 3}),
                     Tensor(p->lengths), p->scale, p->causal);
}

void cleanup(void **metadata) {
    delete reinterpret_cast<PlannedMeta *>(*metadata);
    *metadata = nullptr;
}

static bool registered = [] {
    MultiheadAttention::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MultiheadAttention::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MultiheadAttention::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_impl::ninetoothed_hygon
#endif
