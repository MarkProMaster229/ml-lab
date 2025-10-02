#pragma once

#include "/mnt/storage/product/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Position.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"

// подключаем только заголовок AttentionLogits
#include "AttentionLogits.hpp"

// пересмотрена реализация на более мягкую(уже без хард кода))) )

Tensor matmul(const Tensor& A, const Tensor& B_transposed);
Tensor transpose(const Tensor& X);
Tensor softmax(const Tensor& X);
Tensor operator*(const Tensor& X, const std::vector<float>& W);

class Transformer {
public:
    Transformer(int embedding_dim_, int dk_)
        : embedding_dim(embedding_dim_), dk(dk_), weights() {}

    void load_weights(const std::string& filename);

Tensor forward(Tensor& X) // одна "голова внимания"
{
    Logit::QKV qkv = Logit::complitle(X, weights);
    Tensor scores = matmul(qkv.Q, transpose(qkv.K)) / sqrt(float(dk));
    Tensor attn_weights = softmax(scores);
    Tensor output = matmul(attn_weights, qkv.V);
    return output; // тензор смотрящий на контекст
}

private:
    int embedding_dim;
    int dk;
    Logit::AttentionWeights weights;
};
