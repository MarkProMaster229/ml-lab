#pragma once
#include <vector>
#include <string>
#include "../Generation/Tensor.hpp"

namespace Logit {

struct QKV {
    Tensor Q;
    Tensor K;
    Tensor V;
};

// Класс для хранения весов внимания
class AttentionWeights {
public:
    AttentionWeights();  // конструктор

    bool load(const std::string& filename, int expected_embedding_dim, int expected_dk);

const std::vector<float>& getWq() const;
const std::vector<float>& getWk() const;
const std::vector<float>& getWv() const;
    int embedding_dim;
    int dk;

private:
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;
};

// Функция для вычисления QKV
QKV complitle(const Tensor& X, const AttentionWeights& weights);

// Оператор умножения Tensor * vector
Tensor operator*(const Tensor& X, const std::vector<float>& W);

}
