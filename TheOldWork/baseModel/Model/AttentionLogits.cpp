#include "AttentionLogits.hpp"
#include <iostream>
#include <fstream>
#include <cmath>

namespace Logit {

// Конструктор
AttentionWeights::AttentionWeights() : embedding_dim(0), dk(0) {}

// Загрузка весов из бинарного файла
bool AttentionWeights::load(const std::string& filename, int expected_embedding_dim, int expected_dk) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) {
        std::cerr << "Не удалось открыть файл: " << filename << std::endl;
        return false;
    }

    int file_embedding_dim = 0;
    int file_dk = 0;
    in.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(int));
    in.read(reinterpret_cast<char*>(&file_dk), sizeof(int));

    if (file_embedding_dim != expected_embedding_dim || file_dk != expected_dk) {
        std::cerr << "Размеры в файле не совпадают с ожидаемыми!" << std::endl;
        in.close();
        return false;
    }

    embedding_dim = file_embedding_dim;
    dk = file_dk;

    size_t matrix_size = static_cast<size_t>(embedding_dim) * static_cast<size_t>(dk);
    Wq.resize(matrix_size);
    Wk.resize(matrix_size);
    Wv.resize(matrix_size);

    in.read(reinterpret_cast<char*>(Wq.data()), matrix_size * sizeof(float));
    in.read(reinterpret_cast<char*>(Wk.data()), matrix_size * sizeof(float));
    in.read(reinterpret_cast<char*>(Wv.data()), matrix_size * sizeof(float));

    in.close();
    return true;
}

const std::vector<float>& AttentionWeights::getWq() const { return Wq; }
const std::vector<float>& AttentionWeights::getWk() const { return Wk; }
const std::vector<float>& AttentionWeights::getWv() const { return Wv; }


// Оператор умножения Tensor * vector
Tensor operator*(const Tensor& X, const std::vector<float>& W) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int embedding_dim = X.shape[2];
    int dk = W.size() / embedding_dim;

    Tensor result(batch, seq_len, dk);

    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int k = 0; k < dk; ++k) {
                float val = 0.0f;
                for (int d = 0; d < embedding_dim; ++d) {
                    val += X.at(b, s, d) * W[d * dk + k];
                }
                result.at(b, s, k) = val;
            }
        }
    }

    return result;
}

// Вычисление QKV
QKV complitle(const Tensor& X, const AttentionWeights& weights) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int dk = weights.dk;

    Tensor Q(batch, seq_len, dk);
    Tensor K(batch, seq_len, dk);
    Tensor V(batch, seq_len, dk);

    Q = X * weights.getWq();
    K = X * weights.getWk();
    V = X * weights.getWv();

    return { Q, K, V };

}

} // namespace Logit
