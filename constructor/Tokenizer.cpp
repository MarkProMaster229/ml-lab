#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "Tensor.cpp"
using namespace std;

// далее решаем задачу позиционирования слов для обработки через слой трансформера 
// В трансформере порядок слов важен, потому что сам трансформер не видит порядок.
class PositionalEncoding 
{
    int dim;      // это размерность эмбеддинга
    int max_len;  // максимальная длина последовательности

    /*
    positions — теперь используем Tensor вместо vector<vector<float>>.
    Каждая строка соответствует позиции токена,
    каждый столбец — компоненте позиционного вектора.
    Размерность: [max_len, 1, dim], где batch_size=1.
    */
    Tensor positions;

public:
    // Конструктор создаёт позиции и заполняет их синусами и косинусами
    // то что происходит ниже для меня пока магия но оно работает — у эмбдинга появляется позиция!
    PositionalEncoding(int dim_, int max_len_ = 512) 
        : dim(dim_), max_len(max_len_), positions(max_len_, 1, dim_)
    {
        for (int pos = 0; pos < max_len; ++pos) 
        {
            for (int i = 0; i < dim; ++i) 
            {
                float angle = pos / pow(10000.0f, 2.0f * (i / 2) / dim);
                if (i % 2 == 0)
                    positions.at(pos, 0, i) = sin(angle);
                else
                    positions.at(pos, 0, i) = cos(angle);
            }
        }
    }

    // Метод добавляет позиционные кодировки к эмбеддингам
    // embeddings — Tensor размерности [1, seq_len, dim]
    Tensor add_to_embeddings(const Tensor& embeddings) 
    {
        Tensor out(1, embeddings.seq_len, embeddings.dim); // создаём новый Tensor для выхода
        for (size_t s = 0; s < embeddings.seq_len; ++s) 
        {
            for (int d = 0; d < dim; ++d) 
            {
                // добавляем позицию к эмбеддингу токена
                out.at(0, s, d) = embeddings.at(0, s, d) + positions.at(s, 0, d);
            }
        }
        return out;
    }
    // магия закончилась
};

class Transformer 
{
public:
    // тут тупо базовые манипуляции с матрицами, не более

    // Умножение двух матриц A * B
    // A: [n, m], B: [m, p] → возвращает Tensor [1, n, p] (batch_size = 1)
    Tensor matmul(const Tensor& A, const Tensor& B) {
        size_t n = A.seq_len;      // количество строк в A
        size_t m = A.dim;          // количество столбцов в A
        size_t p = B.dim;          // количество столбцов в B

        Tensor C(1, n, p); // создаём Tensor для результата

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < m; ++k) {
                    sum += A.at(0, i, k) * B.at(0, k, j);
                }
                C.at(0, i, j) = sum;
            }
        }
        return C;
    }

    // Транспонирование матрицы
    // A: [1, n, m] → возвращает Tensor [1, m, n]
    Tensor transpose(const Tensor& A) {
        Tensor T(1, A.dim, A.seq_len); // меняем местами seq_len и dim
        for (size_t i = 0; i < A.seq_len; ++i) {
            for (size_t j = 0; j < A.dim; ++j) {
                T.at(0, j, i) = A.at(0, i, j);
            }
        }
        return T;
    }

    // Softmax по строкам
    // A: [1, n, m] → возвращает Tensor того же размера
    Tensor softmax(const Tensor& A) {
        Tensor S(1, A.seq_len, A.dim);
        for (size_t i = 0; i < A.seq_len; ++i) {
            float max_val = A.at(0, i, 0);
            for (size_t j = 0; j < A.dim; ++j) {
                if (A.at(0, i, j) > max_val) max_val = A.at(0, i, j);
            }

            float sum = 0.0f;
            for (size_t j = 0; j < A.dim; ++j) {
                sum += exp(A.at(0, i, j) - max_val);
            }

            for (size_t j = 0; j < A.dim; ++j) {
                S.at(0, i, j) = exp(A.at(0, i, j) - max_val) / sum;
            }
        }
        return S;
    }

};

