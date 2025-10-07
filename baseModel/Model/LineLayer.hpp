#pragma once
#include <iostream>
#include <cmath>
#include <fstream>
#include <vector>
#include <random>


// нужна матрица весов для линейного слоя после трасформера
/*
она проецирует выходной тензор трасформера размером - dk
в размер словаря(vocab_size)
W_out: vocab_size * dk

b_out — это смещение (bias) для линейного слоя.
b_out: vocab_size
Он добавляется к результату матричного умножения, чтобы сдвинуть логиты.


*/


    //std::vector<float> predict_next_token_probs(const Tensor& transformer_output, int batch_index) {
    //Tensor logits = linear(transformer_output, W_out, b_out);
    //Tensor probs = softmax(logits);
    //return probs[batch_index][transformer_output.shape[1]-1];
//}

//int predict_next_token(const Tensor& transformer_output, int batch_index) {
    //auto probs = predict_next_token_probs(transformer_output, batch_index);
    //return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
//}

//тест!
class LineLayer
{
    private:
    int vocab_size;
    int dk;
    std::vector<float> W_out;
    std::vector<float> b_out;
    public:
    void initialize_or_load(const std::string& filename) {
        if (std::filesystem::exists(filename)) {
            std::ifstream in(filename, std::ios::binary);
            if (!in) {
                std::cerr << "Не удалось открыть файл для чтения: " << filename << std::endl;
                return;
            }

            int file_vocab = 0, file_dk = 0;
            in.read(reinterpret_cast<char*>(&file_vocab), sizeof(int));
            in.read(reinterpret_cast<char*>(&file_dk), sizeof(int));

            if (file_vocab != vocab_size || file_dk != dk) {
                std::cerr << "Размерности файла не совпадают с текущими настройками." << std::endl;
                in.close();
                return;
            }

            in.read(reinterpret_cast<char*>(W_out.data()), W_out.size() * sizeof(float));
            in.read(reinterpret_cast<char*>(b_out.data()), b_out.size() * sizeof(float));
            in.close();

            std::cout << "Файл с весами загружен: " << filename << std::endl;
        } else {
            // Инициализация матриц по Xavier
            float stddev = std::sqrt(2.0f / (dk + vocab_size));
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, stddev);

            for (auto &w : W_out) w = dist(gen);
            for (auto &b : b_out) b = dist(gen);


            save(filename);

            std::cout << "Файл с весами создан и сохранён: " << filename << std::endl;
        }
    }

    // Сохраняем текущие матрицы в файл с заголовком
    void save(const std::string& filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Не удалось создать файл для записи: " << filename << std::endl;
            return;
        }

        // Сначала заголовок: размеры матриц
        out.write(reinterpret_cast<char*>(&vocab_size), sizeof(int));
        out.write(reinterpret_cast<char*>(&dk), sizeof(int));

        // Потом сами данные
        out.write(reinterpret_cast<char*>(W_out.data()), W_out.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(b_out.data()), b_out.size() * sizeof(float));

        out.close();
        std::cout << "Файл с весами сохранён: " << filename << std::endl;
    }
        void initialize(int vocab_size_, int dk_) {
        vocab_size = vocab_size_;
        dk = dk_;

        W_out.resize(vocab_size * dk);
        b_out.resize(vocab_size);

        float stddev = std::sqrt(2.0f / (vocab_size + dk));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, stddev);

        for (auto &w : W_out) w = dist(gen);
        for (auto &b : b_out) b = dist(gen);
    }

    /*
    далее для линейного слоя необходимо сделать взятие
    Tensor output = transformer.forward(final_input); всех пареметров из
    трасформера и прогнать через LineLayer до получения логитов и вероятностей
    далее эти логиты прогнать через - softmax получить сухоую весовую вероятность
    далее сопоставить со словарем - вернуться к  токенизатору благо он посимвольный
    это сильно упростит задачу
    важно!
    linePosition.initialize(vocab_size, dk); - если если ещё не инициализирован
    linePosition.forward(output); - проход

┌───────────────────────────┐
│       LineLayer           │
│  (линейный слой проекции) │
└───────────────────────────┘
        │ logits
        ▼
Логиты:
shape: [batch_size, seq_len, vocab_size]
пример: [3, 24, 260]
логиты = output @ W_out + b_out
W_out: [dk, vocab_size]  (64 × 260)
b_out: [vocab_size]       (260)
        │
        ▼
┌───────────────────────────┐
│         Softmax           │
└───────────────────────────┘
        │ вероятности
        ▼
Probabilities:
shape: [batch_size, seq_len, vocab_size]
пример: [3, 24, 260]
каждый вектор вдоль последнего измерения → вероятности для каждого токена словаря
каждый вектор вдоль последнего измерения -
(batch 0:
seq 0 → vector длиной vocab_size → softmax → probabilities(вероятность)
seq 1 → vector длиной vocab_size → softmax → probabilities
...
seq 23 → vector длиной vocab_size → softmax → probabilities)
        │
        ▼
┌───────────────────────────┐
│       Tokenizer           │
│  (обратное преобразование │
│   индексов в символы)     │
└───────────────────────────┘
        │
        ▼
Output tokens / символы
["hello world", "привет мир", ...]

    */
   void liner(const Tensor& transformer_output, Tensor& out_probs, const std::string& weight_file = "output_layer.pt") {
    int batch_size = transformer_output.shape[0];
    int seq_len    = transformer_output.shape[1];
    int dk         = transformer_output.shape[2];
    int vocab_size = Tokenizer::SIZE;

    // Загружаем веса и bias из файла
    std::vector<float> W_out(vocab_size * dk);
    std::vector<float> b_out(vocab_size);

    std::ifstream in(weight_file, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Не удалось открыть файл с весами: " + weight_file);
    }

    int file_vocab = 0, file_dk = 0;
    in.read(reinterpret_cast<char*>(&file_vocab), sizeof(int));
    in.read(reinterpret_cast<char*>(&file_dk), sizeof(int));


    if (file_vocab != vocab_size || file_dk != dk) {
        in.close();
        throw std::runtime_error("Размерности в файле не совпадают с transformer_output и словарем");
    }

    in.read(reinterpret_cast<char*>(W_out.data()), W_out.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(b_out.data()), b_out.size() * sizeof(float));
    in.close();

    // создаём Tensor под вероятности
    out_probs = Tensor(batch_size, seq_len, vocab_size);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            std::vector<float> logits(vocab_size, 0.0f);
            for (int v = 0; v < vocab_size; v++) {
                for (int k = 0; k < dk; k++)
                    logits[v] += transformer_output.at(i, j, k) * W_out[v * dk + k];
                logits[v] += b_out[v];
            }

            // softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (auto& x : logits) x = std::exp(x - max_logit), sum_exp += x;
            for (int v = 0; v < vocab_size; v++) logits[v] /= sum_exp;

            // записываем в out_probs
            for (int v = 0; v < vocab_size; v++)
                out_probs.at(i, j, v) = logits[v];
        }
    }
}


};