#include "Transformer.hpp"
#include <iostream>
#include <cmath>

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


Transformer::Transformer(int embedding_dim_, int dk_)
    : embedding_dim(embedding_dim_), dk(dk_), weights() {}

// Загрузка весов
void Transformer::load_weights(const std::string& filename) {
    if (!weights.load(filename, embedding_dim, dk)) {
        std::cerr << "Ошибка при загрузке весов!" << std::endl;
    }
}

// Прямой проход (одна голова внимания)
Tensor Transformer::forward(Tensor& X) {
    std::cout << "X shape: [" << X.shape[0] << " " << X.shape[1] << " " << X.shape[2] << "]\n";
    std::cout << "weights: embedding_dim=" << weights.embedding_dim
              << ", dk=" << weights.dk
              << ", Wq.size=" << weights.getWq().size()
              << ", Wk.size=" << weights.getWk().size()
              << ", Wv.size=" << weights.getWv().size()
              << std::endl;

    // Вычисляем Q, K, V
    Logit::QKV qkv = Logit::complitle(X, weights);

    // Attention: softmax(Q K^T / sqrt(dk)) V
    Tensor scores = matmul(qkv.Q, transpose(qkv.K)) / sqrt(float(dk));

    Tensor attn_weights = softmax(scores);
    Tensor output = matmul(attn_weights, qkv.V);

    return output;
}

// Вспомогательные функции - для транспонирования матриц - я  хз как оно работает было украдено в chatGPT, а все что ниже подчистую)))
// да и по*уй))) -  weights: embedding_dim=128, dk=64, Wq.size=8192, Wk.size=8192, Wv.size=8192 матрица прямоугольная(128 x 64) и отлично все что можно было - теперь корректно.
Tensor matmul(const Tensor& A, const Tensor& B) {
    int batch = A.shape[0];
    int M = A.shape[1];
    int K = A.shape[2];
    int N = B.shape[2];

    // Проверка совместимости размерностей
    if (B.shape[1] != K) {
        throw std::runtime_error("matmul: несовпадение размерностей последней оси A и второй оси B");
    }
    Tensor result(batch, M, N);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < K; ++k) {
                    sum += A.at(b, i, k) * B.at(b, k, j);
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

    Tensor Y(batch, dk, seq_len); // shape[2] = seq_len
    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dk; ++j)
                Y.at(b, j, i) = X.at(b, i, j);

    return Y;
}



Tensor softmax(const Tensor& X) {
    Tensor result = X;
    int batch = X.shape[0];
    int rows = X.shape[1];
    int cols = X.shape[2];

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            float max_val = -1e30f;
            for (int j = 0; j < cols; ++j)
                if (X.at(b, i, j) > max_val) max_val = X.at(b, i, j);

            float sum_exp = 0.0f;
            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) = std::exp(X.at(b, i, j) - max_val);
                sum_exp += result.at(b, i, j);
            }

            for (int j = 0; j < cols; ++j)
                result.at(b, i, j) /= sum_exp;
        }
    }

    return result;
}
