#pragma once

#include "/home/chelovek/Документы/work/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/home/chelovek/Документы/work/ml-lab/baseModel/Generation/Position.hpp"
#include "/home/chelovek/Документы/work/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/home/chelovek/Документы/work/ml-lab/baseModel/Generation/Embedding.hpp"
#include "/home/chelovek/Документы/work/ml-lab/baseModel/Model/LineLayer.hpp"
#include "/home/chelovek/Документы/work/ml-lab/baseModel/Core/BatchGenerator.hpp"

#include "AttentionLogits.hpp"


Tensor matmul(const Tensor& A, const Tensor& B_transposed);
Tensor transpose(const Tensor& X);
Tensor softmax(const Tensor& X);
Tensor operator*(const Tensor& X, const std::vector<float>& W);

class Transformer {
public:
    Transformer(int embedding_dim_, int dk_);

    void load_weights(const std::string& filename);
    Tensor forward(Tensor& X);

private:
    int embedding_dim;
    int dk;
    Logit::AttentionWeights weights;
};

std::string transformer_generate(Transformer& transformer, LineLayer& line,
                                 BatchGenerator& batchGen, Tokenizer& tokenizer,
                                 const std::string& input, int steps);