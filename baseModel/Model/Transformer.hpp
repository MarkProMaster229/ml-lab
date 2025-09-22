#pragma once
#include <vector>
#include <string>
#include <fstream>
#include "Tensor.hpp"

class Transformer {
public:
    struct QKV {
        Tensor Q;
        Tensor K;
        Tensor V;
    };

    Transformer(int embedding_dim, int dk);

    void load_weights(const std::string& filename);
    QKV head(Tensor& X);

private:
    int embedding_dim;
    int dk;
    std::vector<float> Wq, Wk, Wv;
};
