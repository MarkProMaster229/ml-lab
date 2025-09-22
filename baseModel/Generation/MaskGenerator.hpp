#pragma once
#include <vector>
#include "Tensor.hpp" // предполагаем, что Tensor.hpp уже есть и содержит класс Tensor
#include <iostream>

class MaskGenerator {
public:
    // Создание маски паддингов для батча токенов
    Tensor createMask(const std::vector<std::vector<int>>& all_tokens) {
        int batch = all_tokens.size();
        int max_seq_len = 0;
        // Находим максимальную длину предложения
        for (auto &tokens : all_tokens)
            if (tokens.size() > max_seq_len)
                max_seq_len = tokens.size();

        Tensor mask(batch, max_seq_len, 1); // один канал на токен

        for (int i = 0; i < batch; i++) {
            int seq_len = all_tokens[i].size();
            for (int j = 0; j < max_seq_len; j++) {
                // 1.0 = токен, 0.0 = padding
                mask.at(i, j, 0) = (j < seq_len) ? 1.0f : 0.0f;
            }
        }

        return mask;
    }

    // Для дебага: печать маски первого батча
    void printMask(const Tensor& mask) {
        std::cout << "Mask preview:" << std::endl;
        for (int j = 0; j < mask.shape[1]; j++) {
            std::cout << mask.at(0, j, 0) << " ";
        }
        std::cout << std::endl;
    }
};
