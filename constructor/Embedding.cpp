#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
using namespace std;
#include "Tokenizer.cpp"
#include "Tensor.cpp"



//размечаем эмбдинги
class Embedding : public Tokenizer
{
public:
    int dim; // размерность эмбеддинга, т.е. сколько чисел будет представлять один токен

    /*
    weights[token_id] — это вектор размером dim для конкретного токена.
    Теперь используем Tensor вместо vector<vector<float>>, чтобы хранить эмбеддинги всех токенов.
    */
    Tensor weights;

    // конструктор: инициализация весов нулями
    Embedding(int dim_) : dim(dim_), weights(Vocab::SIZE, 1, dim_) 
    {
        // по сути нули — неидеально для обучения, лучше использовать случайное распределение
        // например: weights.at(token_id, 0, d) = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
    }

    // Получение эмбеддинга конкретного токена
    // Раньше возвращали vector<float>, теперь просто копируем в Tensor на 1 токен
    Tensor get_embedding(int token_id)
    {
        Tensor emb(1, 1, dim); // 1 батч, 1 токен
        for (int d = 0; d < dim; ++d)
            emb.at(0, 0, d) = weights.at(token_id, 0, d);
        return emb;
    }

    // Кодируем строку в последовательность эмбеддингов
    Tensor encode_to_embedding(const string& text)
    {
        vector<int> token_ids = ByteTokinizer().encode(text);
        Tensor out(1, token_ids.size(), dim); // 1 батч, seq_len = количество токенов

        for (size_t s = 0; s < token_ids.size(); ++s)
        {
            int id = token_ids[s];
            for (int d = 0; d < dim; ++d)
                out.at(0, s, d) = weights.at(id, 0, d);
        }

        return out;
    }
};