#pragma once
#include "../Generation/Position.hpp"
#include "../Generation/Tensor.hpp"
#include "../Generation/MaskGenerator.hpp" // <-- правильное имя
#include "../Generation/Embedding.hpp"
#include <vector>
#include <cstdlib>
#include <ctime>

class BatchGenerator {
public:
    BatchGenerator(int embedding_dim) : embedding_dim(embedding_dim) {
        srand(time(nullptr));
    }

    Tensor createInputTensor(const std::vector<std::vector<int>>& all_tokens);
    Tensor createMask(const std::vector<std::vector<int>>& all_tokens);

private:
    int embedding_dim;
    Position position;
    MaskGenerator mask;
};
