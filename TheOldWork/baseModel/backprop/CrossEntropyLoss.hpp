#pragma once
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/Generation/Tensor.hpp"
#include <vector>
#include <cassert>
#include <cmath>

class Loss {
public:
    Loss() {}

    // вычисление кросс-энтропии
    float forward(const Tensor& probs, const std::vector<int>& y_true) {
        assert(probs.shape[0] == 1 || probs.shape[0] == static_cast<int>(y_true.size() / probs.shape[1]));
        assert(probs.shape[2] > 0);

        float loss = 0.0f;
        int batch = probs.shape[0];
        int seq_len = probs.shape[1];

        for (int b = 0; b < batch; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                int target = y_true[b * seq_len + t];
                float p = probs.at(b, t, target);
                loss -= std::log(p + 1e-9f);
            }
        }

        return loss / (batch * seq_len);
    }

    // градиент по входу softmax
    Tensor backward(const Tensor& probs, const std::vector<int>& y_true) {
        Tensor grad = probs;  //тензор того же размера

        int batch = probs.shape[0];
        int seq_len = probs.shape[1];
        int vocab_size = probs.shape[2];

        for (int b = 0; b < batch; ++b) {
            for (int t = 0; t < seq_len; ++t) {
                int target = y_true[b * seq_len + t];
                for (int v = 0; v < vocab_size; ++v) {
                    grad.at(b, t, v) = probs.at(b, t, v);
                    if (v == target) grad.at(b, t, v) -= 1.0f;
                }
            }
        }

        float scale = 1.0f / (batch * seq_len);
        for (size_t i = 0; i < grad.size(); ++i) grad.data[i] *= scale;

        return grad;
    }
};