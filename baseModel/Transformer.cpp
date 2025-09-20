#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "Tensor.cpp"
using namespace std;

class positioning
{
    public:
    Tensor createPositions(int seq_len, int embedding_dim, int batch = 1)
    {
        Tensor t(batch, seq_len, embedding_dim);
        for (int i = 0; i <batch; i++)
        {
            for (int j = 0; j<seq_len; j++)
            {
                for (int k = 0; k<embedding_dim; k++)
                {
                    //синусо-косинусное кодирование
                    float pos = (float)j;
                    float div_term = pow(10000.0f, 2.0f * (k / 2) / embedding_dim);
                    if (k % 2 == 0)
                        t.at(i, j, k) = sin(pos / div_term);
                    else
                         t.at(i, j, k) = cos(pos / div_term);
                }
            }

        }

        return t;
    }



};

class mask
{
    public:
    Tensor createMask(const std::vector<std::vector<int>>& all_tokens) {
    int batch = all_tokens.size();
    int max_seq_len = 0;
    for (auto &tokens : all_tokens)
        if (tokens.size() > max_seq_len)
            max_seq_len = tokens.size();

    Tensor mask(batch, max_seq_len, 1); // один канал на токен
    for (int i = 0; i < batch; i++) {
        int seq_len = all_tokens[i].size();
        for (int j = 0; j < max_seq_len; j++) {
            mask.at(i, j, 0) = (j < seq_len) ? 1.0f : 0.0f; // 1.0 = токен, 0.0 = padding
        }
    }
    return mask;
}


};


// TODO - вынеси класс - мне не нравится
//ладно
class Transformer
{
    public:
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
struct QKV {
    Tensor Q;
    Tensor K;
    Tensor V;
};

QKV head() {
    Tensor X;
    X.load("tensor.pt");

int batch = X.shape[0];         // batch
int seq_len = X.shape[1];       // seq_len
int embedding_dim = X.shape[2]; // embedding_dim

int dk = embedding_dim;         //размер Q/K/V для одной головы

    Tensor Q(batch, seq_len, dk);
    Tensor K(batch, seq_len, dk);
    Tensor V(batch, seq_len, dk);

    for (int b = 0; b < batch; b++) {
        for (int i = 0; i < seq_len; i++) {
            for (int j = 0; j < dk; j++) {
                float q_val = 0, k_val = 0, v_val = 0;
                for (int k = 0; k < embedding_dim; k++) {
                    q_val += X.at(b, i, k) * Wq.at(k, j);
                    k_val += X.at(b, i, k) * Wk.at(k, j);
                    v_val += X.at(b, i, k) * Wv.at(k, j);
                }
                Q.at(b, i, j) = q_val;
                K.at(b, i, j) = k_val;
                V.at(b, i, j) = v_val;
            }
        }
    }

    return {Q, K, V};
}


};
