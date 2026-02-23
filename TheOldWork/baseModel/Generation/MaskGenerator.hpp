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

/*
суть - перекрыть нули и не подать их в трансформер - для лучшей сходимости
итерируемся по batch и смотрим - мы стоим например на i - 0
0-ой токен, 0-1 ? связи нет  строим матрицу - mask.at(b, i, j) = 0.0f;
дальше например мы на 4 токене, можем сделать 4 - 1 ? или 4- 2 ?
да можем, связь между токенами есть все нормально
дальше например встертили конец max_seq_len очевидно что дальше пудут только padding(нули)
все дальше связи между токенами не будет
Токен i может собирать информацию от всех предыдущих токенов и себя, но не из будущего и не из padding


*/
        Tensor mask(batch, max_seq_len, max_seq_len); // треугольная + padding
        for (int b = 0; b < batch; b++) {
            int seq_len = all_tokens[b].size(); // реальная длина последовательности
            for (int i = 0; i < max_seq_len; i++) {
                for (int j = 0; j < max_seq_len; j++) {
                    if (i < seq_len && j < seq_len && j <= i) {
                        mask.at(b, i, j) = 1.0f; // токен i может смотреть на токен j
                    }
                else {
                    mask.at(b, i, j) = 0.0f; // будущее или padding
                }
        }
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
