// задача собрать логиты(грубо говоря сырье) для механизма внимания
// как ?
/*
матрицы логитов формируются по формуле
scores = (Q * K^T) / sqrt(dk)
откуда взять QKV?
из матриц внимания по -
Wq ∈ ℝ^(embedding_dim × dk)

Wk ∈ ℝ^(embedding_dim × dk)

Wv ∈ ℝ^(embedding_dim × dk)
далее пропустить через softmax(я не знаю как пока что)
и в теории домножить на V - получим итог внимания ну чисто в теории

*/
#include <iostream>
#include <vector>
#include <string>
#include <fstream>

class Logit
{
    public:

struct AttentionWeights {
    int embedding_dim;
    int dk;
    std::vector<float> Wq;
    std::vector<float> Wk;
    std::vector<float> Wv;

    // Конструктор пустой
    AttentionWeights() : embedding_dim(0), dk(0) {}

    // Загружаем веса из файла с заголовком
    bool load(const std::string& filename, int expected_embedding_dim, int expected_dk) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "Не удалось открыть файл: " << filename << std::endl;
            return false;
        }

        // Читаем заголовок
        int file_embedding_dim = 0;
        int file_dk = 0;
        in.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(int));
        in.read(reinterpret_cast<char*>(&file_dk), sizeof(int));

        // Проверяем размеры
        if (file_embedding_dim != expected_embedding_dim || file_dk != expected_dk) {
            std::cerr << "Размеры в файле не совпадают с ожидаемыми!" << std::endl;
            std::cerr << "Файл: embedding_dim=" << file_embedding_dim
                      << ", dk=" << file_dk
                      << " | Ожидается: embedding_dim=" << expected_embedding_dim
                      << ", dk=" << expected_dk << std::endl;
            in.close();
            return false;
        }

        embedding_dim = file_embedding_dim;
        dk = file_dk;

        // Вычисляем размеры каждой матрицы
        size_t matrix_size = static_cast<size_t>(embedding_dim) * static_cast<size_t>(dk);

        Wq.resize(matrix_size);
        Wk.resize(matrix_size);
        Wv.resize(matrix_size);

        // Читаем данные
        in.read(reinterpret_cast<char*>(Wq.data()), matrix_size * sizeof(float));
        in.read(reinterpret_cast<char*>(Wk.data()), matrix_size * sizeof(float));
        in.read(reinterpret_cast<char*>(Wv.data()), matrix_size * sizeof(float));

        in.close();
        return true;
    }

    // Геттеры для матриц
    std::vector<float>& getWq() { return Wq; }
    std::vector<float>& getWk() { return Wk; }
    std::vector<float>& getWv() { return Wv; }
};



};