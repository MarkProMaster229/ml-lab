#include <iostream>
#include <vector>
#include <string>
#include <fstream>


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



class Transformer
{
    public:


};
