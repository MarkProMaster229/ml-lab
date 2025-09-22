#include "BatchGenerator.hpp"          // Объявление класса
#include "../Generation/Position.hpp"
#include "../Generation/Tensor.hpp"
#include "../Generation/MaskGenerator.hpp"
#include <cstdlib>
#include <ctime>

Tensor BatchGenerator::createInputTensor(const std::vector<std::vector<int>>& all_tokens) {
    int batch = all_tokens.size();
    int max_seq_len = 0;
    for (auto &tokens : all_tokens)
        if (tokens.size() > max_seq_len)
            max_seq_len = tokens.size();

    Tensor token_embeddings(batch, max_seq_len, this->embedding_dim);

    for (int i = 0; i < batch; i++) {
        int seq_len = all_tokens[i].size();
        for (int j = 0; j < max_seq_len; j++) {
            for (int k = 0; k < this->embedding_dim; k++) {
                token_embeddings.at(i, j, k) = (j < seq_len) ? static_cast<float>(rand()) / RAND_MAX : 0.0f;
            }
        }
    }

    Tensor pos_encoding = this->position.createPositions(max_seq_len, this->embedding_dim, batch);
    Tensor final_input(batch, max_seq_len, this->embedding_dim);

    for (int i = 0; i < batch; i++)
        for (int j = 0; j < max_seq_len; j++)
            for (int k = 0; k < this->embedding_dim; k++)
                final_input.at(i, j, k) = token_embeddings.at(i, j, k) + pos_encoding.at(i, j, k);

    return final_input;
}

Tensor BatchGenerator::createMask(const std::vector<std::vector<int>>& all_tokens) {
    return this->mask.createMask(all_tokens);
}
