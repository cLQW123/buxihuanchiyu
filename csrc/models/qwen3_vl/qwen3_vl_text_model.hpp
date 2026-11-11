#pragma once

#include "../../layers/causal_lm_templates/text_decoder_layer.hpp"
#include "../../layers/common_modules.hpp"
#include "qwen3_vl_attention.hpp"

#include <infinicore/nn/embedding.hpp>
#include "../../layers/normalization/qwen_rms_ninetoothed.hpp"
#include <infinicore/nn/rmsnorm.hpp>
#include <infinicore/tensor.hpp>

#include <memory>
#include <vector>

namespace infinilm::models::qwen3_vl {

using Qwen3VLMLP = infinilm::layers::MLP;
class Qwen3VLDecoderLayer : public infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3VLAttention,Qwen3VLMLP> {
 using Base = infinilm::layers::causal_lm_templates::TextDecoderLayer<Qwen3VLAttention,Qwen3VLMLP>;
public:
 using Base::Base;
 std::tuple<infinicore::Tensor,infinicore::Tensor> forward(const infinicore::Tensor &positions, infinicore::Tensor &hidden_states, infinicore::Tensor &residual) {
  residual = residual ? infinicore::op::add(residual,hidden_states) : hidden_states;
  hidden_states = infinilm::layers::normalization::qwen_rms(residual,this->input_layernorm_);
  hidden_states = this->self_attn_->forward(positions,hidden_states);
  residual = infinicore::op::add(residual,hidden_states);
  hidden_states = infinilm::layers::normalization::qwen_rms(residual,this->post_attention_layernorm_);
  hidden_states = this->mlp_->forward(hidden_states);
  return {hidden_states,residual};
 }
};

struct Qwen3VLDeepStackEmbedding {
    infinicore::Tensor features;
    size_t token_offset;
    size_t decoder_layer;
};

class Qwen3VLTextModel : public infinicore::nn::Module {
public:
    Qwen3VLTextModel(std::shared_ptr<infinilm::config::ModelConfig> model_config,
                     const infinicore::Device &device);

    infinicore::Tensor forward(const infinilm::InfinilmModel::Input &input) const;

    infinicore::Tensor forward_embeds(const infinicore::Tensor &inputs_embeds,
                                      const infinicore::Tensor &position_ids,
                                      const std::vector<Qwen3VLDeepStackEmbedding>
                                          &deepstack_embeddings
                                      = {}) const;

    infinicore::Tensor embed_tokens(const infinicore::Tensor &input_ids) const;

private:
    infinicore::Tensor run_layers(
        infinicore::Tensor hidden_states, const infinicore::Tensor &position_ids,
        const std::vector<Qwen3VLDeepStackEmbedding> &deepstack_embeddings) const;

    INFINICORE_NN_MODULE(infinicore::nn::Embedding, embed_tokens);
    INFINICORE_NN_MODULE_VEC(Qwen3VLDecoderLayer, layers);
    INFINICORE_NN_MODULE(infinicore::nn::RMSNorm, norm);

    bool skip_final_norm_{false};
};

} // namespace infinilm::models::qwen3_vl
