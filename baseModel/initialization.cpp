#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include "Tokenizer.cpp"
#include "Tensor.cpp"
#include "Transformer.cpp"
class initialization {
public:

    // Генерация случайного тензора токенов с последующим сложением с позиционным кодированием
    Tensor freeRandom(const std::vector<std::vector<int>>& all_tokens, int embedding_dim)
    {
        srand(time(nullptr));

        int batch = all_tokens.size();

        // Находим максимальную длину предложения
        int max_seq_len = 0;
        for (auto &tokens : all_tokens)
            if (tokens.size() > max_seq_len)
                max_seq_len = tokens.size();

        // Создаем тензор для токенов
        Tensor token_embeddings(batch, max_seq_len, embedding_dim);

        for (int i = 0; i < batch; i++) {
            int seq_len = all_tokens[i].size();
            for (int j = 0; j < max_seq_len; j++) {
                for (int k = 0; k < embedding_dim; k++) {
                    if (j < seq_len)
                        token_embeddings.at(i, j, k) = (float)rand() / RAND_MAX;
                    else
                        token_embeddings.at(i, j, k) = 0.0f; // PAD
                }
            }
        }

        // Позиционное кодирование
        positioning pos;
        Tensor pos_encoding = pos.createPositions(max_seq_len, embedding_dim, batch);

        // Суммируем поэлементно
        Tensor final_input(batch, max_seq_len, embedding_dim);
        for (int i = 0; i < batch; i++)
            for (int j = 0; j < max_seq_len; j++)
                for (int k = 0; k < embedding_dim; k++)
                    final_input.at(i, j, k) = token_embeddings.at(i, j, k) + pos_encoding.at(i, j, k);

        return final_input;
    }

    // Главный метод run() для генерации батча и маски
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

        // Генерируем финальный вход для трансформера
        Tensor final_input = freeRandom(all_tokens, embedding_dim);

        // Генерируем маску для паддингов
        mask m;
        Tensor mask_tensor = m.createMask(all_tokens);

        // Для проверки
        std::cout << "Final input shape: ["
                  << final_input.shape[0] << " "
                  << final_input.shape[1] << " "
                  << final_input.shape[2] << "]" << std::endl;

        std::cout << "Mask shape: ["
                  << mask_tensor.shape[0] << " "
                  << mask_tensor.shape[1] << " "
                  << mask_tensor.shape[2] << "]" << std::endl;

        // Пример вывода первого элемента
        std::cout << "final_input[0][0][0] = " << final_input.at(0,0,0) << std::endl;
        std::cout << "mask[0][0][0] = " << mask_tensor.at(0,0,0) << std::endl;
    }

};