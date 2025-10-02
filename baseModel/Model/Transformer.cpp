#include "Transformer.hpp"
#include <iostream>
#include <cmath>

// подключаем только заголовочные файлы
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Position.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"
#include "AttentionLogits.hpp"

    /*
    у нас есть эмбединги в тензоре так ?
    сам тензор в pt файле по верному если
    есть - синусо-косинус позиционка
    если брать работу нам нужно реализовать Multi-Head Attention,как это сделать ?
    каждая голова будет считать Q = x *wq, K = X * Wk, V = X*Wv
    и считаем Attention по формуле (Q,K,V) = softmax(QKᵀ / √d_k) V
    затем все головы конкатятся и умножаются на Wo
    так разберем переменные
    X = входные эмбединги(токены + позиционка)
    матрицы преобразования Wq, Wk, Wv -
    Wq - превращает векторы в запросы
    Wk - превращает векторы в ключи
    Wv - превращает векторы в значения
    по сути мы из одного вектора для каждого токена делаем три разных представления -
    Q - что я ищю? , K - что у меня есть ? V - какая у меня есть информация?
    далее самое интересное - механизм внимания -
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) * V
    QKᵀ - умножаем запросы на ключи - это дает таблицу сходств(насколько каждый токен похож на другой)
    / √d_k - делим на корень из рамера ключа - что бы значения не были слишком похожи(иначе softmax взорвется
    softmax(...) - превращаем в вероятности - каждое число - показывает - с какой вероятностью
    токен i должен смотреть на токен j
    (softmax(...) * V) - берем среднее от значения V, взвешенное этими вероятностями -
    то есть каждый токен собирает новую информацию из всех остальных токенов

    теперь то что было бы хорошо сделать -
    Multi-Head Attention - оно же многоголовое внимание
    формально формула выглядит вот так -
    head_i = Attention(X * Wq_i, X * Wk_i, X * Wv_i)

    пока делаем одну голову дальше смотрим по сходимости или просто потом решу




    */
// пересмотрена реализация на более мягкую(уже без хард кода))) )

Tensor matmul(const Tensor& A, const Tensor& B_transposed) {
    int batch = A.shape[0];
    int seq_len = A.shape[1];
    int dim = A.shape[2]; // размерность Q/K

    Tensor result(batch, seq_len, seq_len); // внимание квадратное

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < dim; ++k) {
                    sum += A.at(b, i, k) * B_transposed.at(b, j, k); // B уже транспонировано
                }
                result.at(b, i, j) = sum;
            }
        }
    }
    return result;
}

Tensor transpose(const Tensor& X) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int dk = X.shape[2];

    Tensor Y(batch, dk, seq_len);

    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dk; ++j)
                Y.at(b, j, i) = X.at(b, i, j);

    return Y;
}


Tensor softmax(const Tensor& X) {
    Tensor result = X; // создаём копию

    int batch = X.shape[0];
    int rows = X.shape[1];
    int cols = X.shape[2];

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            float max_val = -1e30f;
            for (int j = 0; j < cols; ++j) {
                if (X.at(b, i, j) > max_val) max_val = X.at(b, i, j);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) = std::exp(X.at(b, i, j) - max_val); // для стабильности
                sum_exp += result.at(b, i, j);
            }

            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) /= sum_exp;
            }
        }
    }

    return result;
}
void Transformer::load_weights(const std::string& filename) {
    if (!weights.load(filename, embedding_dim, dk)) {
        std::cerr << "Ошибка при загрузке весов!" << std::endl;
    }
}
