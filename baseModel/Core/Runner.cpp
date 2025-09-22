#include "Runner.hpp"

#include "../Generation/Position.hpp"
#include "../Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"
Runner::Runner(int embedding_dim, int dk)
    : embedding_dim(embedding_dim), dk(dk), transformer(embedding_dim, dk), batchGen(embedding_dim) {}

void Runner::run() {
    Embedding em;
    auto all_tokens = em.GetAnaliz();
    if (all_tokens.empty()) {
        std::cerr << "Нет предложений для обработки!" << std::endl;
        return;
    }

    std::string tensor_file = "tensor.pt";
    Tensor final_input;

    if (fs::exists(tensor_file)) {
        std::cout << "Файл tensor.pt найден, загружаем существующий тензор." << std::endl;
        final_input.load(tensor_file);
    } else {
        std::cout << "Файл tensor.pt не найден, создаём новый тензор." << std::endl;
        final_input = batchGen.createInputTensor(all_tokens);
        final_input.save(tensor_file);
    }

    Tensor mask_tensor = batchGen.createMask(all_tokens);

    std::cout << "Final input shape: ["
              << final_input.shape[0] << " "
              << final_input.shape[1] << " "
              << final_input.shape[2] << "]" << std::endl;
    std::cout << "Mask shape: ["
              << mask_tensor.shape[0] << " "
              << mask_tensor.shape[1] << " "
              << mask_tensor.shape[2] << "]" << std::endl;
}
