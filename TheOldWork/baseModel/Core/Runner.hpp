#pragma once
#include "../Model/Transformer.hpp"
#include "BatchGenerator.hpp"
#include "../Generation/Embedding.hpp"
#include <filesystem>
#include <iostream>


namespace fs = std::filesystem;

class Runner {
public:
    Runner(int embedding_dim, int dk);
    void run();

private:
    int embedding_dim;
    int dk;
    Transformer transformer;
    BatchGenerator batchGen;
};
