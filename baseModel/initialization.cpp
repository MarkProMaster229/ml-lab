#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "Tokenizer.cpp"
#include "Tensor.cpp"

class initialization
{
public:
    // метод генерации случайного тензора
    Tensor freeRandom(const std::vector<int>& tokens, int embedding_dim)
    {
        srand(time(nullptr));

        int batch = 4;
        int seq_len = tokens.size();

        Tensor ten(batch, seq_len, embedding_dim);

    for (int i = 0; i < batch; i++) {          // цикл по предложениям
        for (int j = 0; j < seq_len; j++) {    // цикл по токенам
            for (int k = 0; k < embedding_dim; k++) {
                ten.at(i, j, k) = (float)rand() / RAND_MAX;
            }
        }
    }
        return ten;
    }

    void run()
    {
        Embeding em;
        std::vector<std::vector<int>> all_tokens = em.GetAnaliz();

        int embedding_dim = 10;
        Tensor t = freeRandom(all_tokens[0], embedding_dim);

        t.save("tensor.pt");

        Tensor t2;
        t2.load("tensor.pt");

        std::cout << "t2[0][0][0] = " << t2.at(0, 0, 0) << std::endl;
    }
};
