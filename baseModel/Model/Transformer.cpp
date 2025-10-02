#include "Transformer.hpp"
#include <iostream>
#include <cmath>

// Конструктор
Transformer::Transformer(int embedding_dim_, int dk_)
    : embedding_dim(embedding_dim_), dk(dk_), weights() {}

// Загрузка весов
void Transformer::load_weights(const std::string& filename) {
    if (!weights.load(filename, embedding_dim, dk)) {
        std::cerr << "Ошибка при загрузке весов!" << std::endl;
    }
}

// Прямой проход (одна голова внимания)
Tensor Transformer::forward(Tensor& X) {
    std::cout << "X shape: [" << X.shape[0] << " " << X.shape[1] << " " << X.shape[2] << "]\n";
    std::cout << "weights: embedding_dim=" << weights.embedding_dim
              << ", dk=" << weights.dk
              << ", Wq.size=" << weights.getWq().size()
              << ", Wk.size=" << weights.getWk().size()
              << ", Wv.size=" << weights.getWv().size()
              << std::endl;

    // Вычисляем Q, K, V
    Logit::QKV qkv = Logit::complitle(X, weights);

    // Attention: softmax(Q K^T / sqrt(dk)) V
    Tensor scores = matmul(qkv.Q, transpose(qkv.K)) / sqrt(float(dk));

    Tensor attn_weights = softmax(scores);
    Tensor output = matmul(attn_weights, qkv.V);

    return output;
}

// Вспомогательные функции - для транспонирования матриц я хз как оно работает было украдено в chatGPT все что ниже подчистую
Tensor matmul(const Tensor& A, const Tensor& B_transposed) {
    int batch = A.shape[0];
    int seq_len = A.shape[1];
    int dk = A.shape[2]; // Q/K dimension

    Tensor result(batch, seq_len, seq_len);

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < dk; ++k) {
                    sum += A.at(b, i, k) * B_transposed.at(b, k, j); // <-- вот здесь меняем
                }
                result.at(b, i, j) = sum;
            }
        }
    }

    return result;
}


Tensor transpose(const Tensor& X) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int dk = X.shape[2];

    Tensor Y(batch, dk, seq_len); // shape[2] = seq_len
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dk; ++j)
                Y.at(b, j, i) = X.at(b, i, j);

    return Y;
}



Tensor softmax(const Tensor& X) {
    Tensor result = X;
    int batch = X.shape[0];
    int rows = X.shape[1];
    int cols = X.shape[2];

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            float max_val = -1e30f;
            for (int j = 0; j < cols; ++j)
                if (X.at(b, i, j) > max_val) max_val = X.at(b, i, j);

            float sum_exp = 0.0f;
            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) = std::exp(X.at(b, i, j) - max_val);
                sum_exp += result.at(b, i, j);
            }

            for (int j = 0; j < cols; ++j)
                result.at(b, i, j) /= sum_exp;
        }
    }

    return result;
}
