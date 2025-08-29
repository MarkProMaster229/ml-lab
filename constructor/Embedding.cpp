#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
using namespace std;
#include "Tokenizer.cpp"


// размечаем эмбдинги
class Embedding : public Tokenizer
{
public:
    int dim; // это размерность эмбеддинга, т.е. сколько чисел будет представлять один токен.
    /*
    weights[token_id] — это вектор размером dim для конкретного токена.
    vector<vector<float>> — двумерный вектор: строки — это токены, столбцы — параметры/веса (эмбеддинги).
    */
    vector<vector<float>> weights;

    // можно менять размер матрицы весов ради прикола можно поставить 512 но не рекомендую
    Embedding(int dim_) : dim(dim_) 
    {
        weights.resize(Vocab::SIZE, vector<float>(dim, 0.0f)); // инициализация нулями - но по сути это не совсем верно 
        // обьясняю почему - при обучении намного легче будет корректирваоть ошибку методом обратного распространения 
        // если веса будут изначально разбросаны, в ином случае все наши слова равны по смыслу с начала
    }

    // Получение эмбеддинга конкретного токена
    // Взял токен по его id и вернул вектор эмбеддинга этого токена.
    // Пример: токен 'h' = 104 → вернётся weights[104], который размерностью dim.
    vector<float> get_embedding(int token_id)
    {
        return weights[token_id];
    }

    /*
    Сначала мы токенизируем текст через ByteTokinizer(), получаем список токенов.
    Потом для каждого токена берём его эмбеддинг через get_embedding.
    В итоге получаем матрицу эмбеддингов размерности [кол-во токенов, dim].
    */
    vector<vector<float>> encode_to_embedding(const string& text)
    {
        vector<int> token_ids = ByteTokinizer().encode(text);
        vector<vector<float>> out;
        for(int id : token_ids) out.push_back(get_embedding(id));
        return out;
    }
};

// далее решаем задачу позиционирования слов для обработки через слой трансформера 
// В трансформере порядок слов важен, потому что сам трансформер не видит порядок.
class PositionalEncoding 
{
    int dim; // это размерность эмбеддинга
    int max_len; // максимальная длина последовательности

    /*
    positions — это двумерный массив, в котором каждая строка соответствует позиции токена,
    а каждый столбец — компоненте позиционного вектора.
    Размерность: [max_len][dim].
    */
    vector<vector<float>> positions;

public:
    // то что происходит ниже для меня пока магия но оно рабоает - у эмбдинга появляется позиция!
    PositionalEncoding(int dim_, int max_len_ = 512) : dim(dim_), max_len(max_len_) 
    {
        positions.resize(max_len, vector<float>(dim, 0.0f));
        for (int pos = 0; pos < max_len; ++pos) 
        {
            for (int i = 0; i < dim; ++i) 
            {
                positions[pos][i] = pos / pow(10000.0, 2.0 * (i / 2) / dim);
                if (i % 2 == 0)
                    positions[pos][i] = sin(positions[pos][i]);
                else
                    positions[pos][i] = cos(positions[pos][i]);
            }
        }
    }

    vector<vector<float>> add_to_embeddings(const vector<vector<float>>& embeddings) 
    {
        vector<vector<float>> out = embeddings;
        for (size_t i = 0; i < embeddings.size(); ++i) 
        {
            for (int j = 0; j < dim; ++j) 
            {
                out[i][j] += positions[i][j];
            }
        }
        return out;
    }
    // магия закончилась 
};
