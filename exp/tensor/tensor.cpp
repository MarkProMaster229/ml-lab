//btw i using tensor code in Rust! cool ?
//but for what ?

//realization autograd in tensor class
#include <vector> 
#include <cstddef>

class Tensor 
{
    size_t embedding_dim;
    std::vector<float>vec;
    std::size_t count;
public:

    Tensor(std::vector<float>vec, std::size_t count,size_t embedding_dim)
    {
        this->vec = vec; 
        this->count = count;
        this->embedding_dim = embedding_dim;
    }

    std::vector<float> embeddingLookup(size_t token)
    {
        std::vector<float> result(embedding_dim);
        for (size_t i = 0; i < embedding_dim; i++)
        {
            //вот так будет выглядеть моя структура внутри тензора 
            result[i] = vec[token * embedding_dim + i];
        }
    }
    

    class Autograd
    {
private:

    };


};