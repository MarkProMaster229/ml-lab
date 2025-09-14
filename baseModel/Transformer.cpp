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



class Transformer
{
    public:


};
