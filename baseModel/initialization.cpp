#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include "Tokenizer.cpp"
#include "Tensor.cpp"
#include "Transformer.cpp"

class initialization
{
public:
    // метод генерации случайного тензора для всех предложений
    Tensor freeRandom(const std::vector<std::vector<int>>& all_tokens, int embedding_dim)
    {
        srand(time(nullptr));

        int batch = all_tokens.size();

        // Находим максимальную длину предложения
        int max_seq_len = 0;
        for (auto &tokens : all_tokens)
            if (tokens.size() > max_seq_len)
                max_seq_len = tokens.size();

        // Создаем тензор [batch, max_seq_len, embedding_dim]
        Tensor ten(batch, max_seq_len, embedding_dim);

        // Заполняем эмбеддинги случайными числами, добавляем паддинг 0 для коротких предложений
        for (int i = 0; i < batch; i++) {
            int seq_len = all_tokens[i].size();
            for (int j = 0; j < max_seq_len; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    if (j < seq_len)
                        ten.at(i, j, k) = (float)rand() / RAND_MAX;
                    else
                        ten.at(i, j, k) = 0.0f; // PAD
                }
            }
        }

        return ten;
    }

    void run()
    {
        Embeding em;
        std::vector<std::vector<int>> all_tokens = em.GetAnaliz();

        if (all_tokens.empty()) {
            std::cerr << "Нет предложений для обработки!" << std::endl;
            return;
        }

        int embedding_dim = 10;

        // Генерируем тензор для всех предложений
        Tensor t = freeRandom(all_tokens, embedding_dim);

        // Сохраняем и загружаем для проверки
        t.save("tensor.pt");

        Tensor t2;
        t2.load("tensor.pt");

        std::cout << "t2[0][0][0] = " << t2.at(0, 0, 0) << std::endl;
        std::cout << "Shape: [ " << t2.shape[0] << " " << t2.shape[1] << " " << t2.shape[2] << " ]" << std::endl;
        int max_seq_len = 0;
        for (auto &tokens : all_tokens)
           if (tokens.size() > max_seq_len)
               max_seq_len = tokens.size();
        int batch = all_tokens.size();
        positioning pos;
        pos.createPositions(max_seq_len,embedding_dim, batch);

// Генерируем позиционные вектора для всех предложений
Tensor pos_enc = pos.createPositions(max_seq_len, embedding_dim, batch);

// Печатаем позиционные вектора для первого элемента батча
for (int j = 0; j < max_seq_len; j++) {
    std::cout << "Pos " << j << ": ";
    for (int k = 0; k < embedding_dim; k++) {
        std::cout << pos_enc.at(0, j, k) << " ";
    }
    std::cout << std::endl;
}


    }
};
