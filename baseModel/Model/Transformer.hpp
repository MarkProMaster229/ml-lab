#pragma once

#include "/mnt/storage/product/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Position.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"

// подключаем только заголовок AttentionLogits
#include "AttentionLogits.hpp"

Tensor matmul(const Tensor& A, const Tensor& B_transposed);
Tensor transpose(const Tensor& X);
Tensor softmax(const Tensor& X);
Tensor operator*(const Tensor& X, const std::vector<float>& W);

class Transformer {
public:
    Transformer(int embedding_dim_, int dk_);

    void load_weights(const std::string& filename);
    Tensor forward(Tensor& X); // объявление функции

private:
    int embedding_dim;
    int dk;
    Logit::AttentionWeights weights;
};