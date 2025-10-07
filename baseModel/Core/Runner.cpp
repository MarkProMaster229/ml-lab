#include "Runner.hpp"

#include "../Generation/Position.hpp"
#include "../Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/WeightGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Model/LineLayer.hpp"
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
    std::string weights_file = "weights.pt";

    WeightGenerator wg(embedding_dim, dk);
    if (fs::exists(weights_file)) {
        std::cout << "weights.pt найден, загружаем..." << std::endl;
        wg.load(weights_file);
    }
    else {
        std::cout << "weights.pt не найден, генерируем..." << std::endl;
        wg.initialize();
        wg.save(weights_file);
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

//TODO - дебаг, проверка работы головы внимания

/*

Final input shape: [3 32 128]

Q shape: [3 32 64]
K shape: [3 32 64]
V shape: [3 32 64]

рассмотрим то что мы получили на выходе
почему все параметры повторяются ?
   потому что у нас только одна голова внимания
что такое 3?
   это batch size - кол-во последовательностей прогоняемых через трансформер
что такое 32?
   это seq_len - длинна последовательности
что такое 64?
   это dk напомню - dk будет являться половиной от embedding_dim

вид маски(маска по Causal / Autoregressive ) сейчас -
Mask shape: [3 32 32]
квадратная матрица 32X32

*/

/*
07.10.2025
linePosition.initialize(128, 64); - постоянные парамеры embedding_dim и dk

итог Transformer output dk: 64, LineLayer dk: 64

норма
Final input shape: [3 24 128]
Mask shape: [3 24 24]
X shape: [3 24 128]


*/
Transformer transformer(embedding_dim, dk);
transformer.load_weights("weights.pt");

// Пропускаем через голову внимания
Tensor output = transformer.forward(final_input);

//линейный слой
LineLayer linePosition;
linePosition.initialize(Tokenizer::SIZE, dk);  // vocab_size = размер словаря (например, 260), dk = 128
linePosition.initialize_or_load("output_layer.pt");

// Получаем вероятности
Tensor probs;
linePosition.liner(output, probs, "output_layer.pt");

// Проверяем форму
std::cout << "Probs shape: ["
          << probs.shape[0] << " "
          << probs.shape[1] << " "
          << probs.shape[2] << "]\n";


// Проверяем, что dk совпадает с ожидаемым
std::cout << "Transformer output dk: " << output.shape[2]
          << ", LineLayer dk: " << dk << "\n";


//тест прямого прохода

          // создаём генератор батчей и токенизатор
BatchGenerator batchGen(embedding_dim);
Tokenizer tokenizer;

// исходный текст
std::string input_text = "при";

// количество шагов генерации
int steps = 10;

// вызов функции генерации
std::string generated_text = transformer_generate(transformer, linePosition, batchGen, tokenizer, input_text, steps);

// вывод результата
std::cout << "Generated text: " << generated_text << std::endl;


std::ofstream fout("output.txt");
fout << generated_text;
fout.close();




// конец теста прямого прохода



//TODO - конец дебаг строк!

}
