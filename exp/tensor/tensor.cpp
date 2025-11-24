//btw i using tensor code in Rust! cool ?
//but for what ?

//realization autograd in tensor class
#include <vector> 
#include <cstddef>

class Tensor 
{
    std::vector<float>vec;
    std::size_t count;
public:

    Tensor(std::vector<float>vec, std::size_t count)
    {
        this->vec = vec; 
        this->count = count;
    
    }

    //std::vector embeddingLookup()
    //{
    
    //}
    

    class Autograd
    {
private:

    };


};