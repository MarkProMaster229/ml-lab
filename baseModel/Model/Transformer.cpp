#include "Transformer.hpp"
#include <iostream>
#include <cmath>

#include "../Generation/Tensor.hpp"
#include "../Generation/Position.hpp"
#include "../Generation/MaskGenerator.hpp"
#include "../Generation/Embedding.hpp"

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

Transformer::Transformer(int embedding_dim, int dk)
    : embedding_dim(embedding_dim), dk(dk),
      Wq(embedding_dim * dk),
      Wk(embedding_dim * dk),
      Wv(embedding_dim * dk) {}

void Transformer::load_weights(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    in.read(reinterpret_cast<char*>(Wq.data()), Wq.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(Wk.data()), Wk.size() * sizeof(float));
    in.read(reinterpret_cast<char*>(Wv.data()), Wv.size() * sizeof(float));
    in.close();
}

Transformer::QKV Transformer::head(Tensor& X) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int embedding_dim = X.shape[2];

    Tensor Q(batch, seq_len, dk);
    Tensor K(batch, seq_len, dk);
    Tensor V(batch, seq_len, dk);

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < dk; j++) {
                float q_val = 0, k_val = 0, v_val = 0;
                for (int k = 0; k < embedding_dim; k++) {
                    q_val += X.at(b, i, k) * Wq[k * dk + j];
                    k_val += X.at(b, i, k) * Wk[k * dk + j];
                    v_val += X.at(b, i, k) * Wv[k * dk + j];
                }
                Q.at(b, i, j) = q_val;
                K.at(b, i, j) = k_val;
                V.at(b, i, j) = v_val;
            }
        }
    }

    return {Q, K, V};
}
