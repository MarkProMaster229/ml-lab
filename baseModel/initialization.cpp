#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

class initialization
{
public:
    Tensor freeRandom(const std::vector<int>& tokens, int embedding_dim)
    {
        srand(time(nullptr)); // разные случайные числа каждый запуск

        int batch = 1;                // одно предложение
        int seq_len = tokens.size();  // сколько токенов в предложении

        Tensor ten(batch, seq_len, embedding_dim);

        for (int j = 0; j < seq_len; j++) {
            for (int k = 0; k < embedding_dim; k++) {
                ten.at(0, j, k) = (float)rand() / RAND_MAX;
            }
        }

        return ten;
    }
};
