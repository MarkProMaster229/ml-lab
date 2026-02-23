#pragma once
#include <cmath>
#include "Tensor.hpp" // Подключаем твой Tensor.hpp

class Position {
public:
    // Создание синусо-косинусного позиционного кодирования
    Tensor createPositions(int seq_len, int embedding_dim, int batch = 1) {
        Tensor t(batch, seq_len, embedding_dim);
        for (int i = 0; i < batch; i++) {
            for (int j = 0; j < seq_len; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    float pos = static_cast<float>(j);
                    float div_term = std::pow(10000.0f, 2.0f * (k / 2) / embedding_dim);
                    if (k % 2 == 0)
                        t.at(i, j, k) = std::sin(pos / div_term);
                    else
                        t.at(i, j, k) = std::cos(pos / div_term);
                }
            }
        }
        return t;
    }
};