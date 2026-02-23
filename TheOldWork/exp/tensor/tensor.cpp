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
        return result;
    }

    std::vector<float> Batch(const std::vector<size_t>& tokens)
    {
        std::vector<float> batch(tokens.size() * embedding_dim);

        for (size_t i = 0; i < tokens.size(); i++)
        {
            std::vector<float> emb = embeddingLookup(tokens[i]);
            for(size_t j = 0; j < embedding_dim; j++)
            {
                batch[i * embedding_dim + j] = emb[j];
            }
        }
        return batch;
    }

    std::vector<std::vector<float>> createBatches(const std::vector<size_t>& all_tokens, size_t batch_size) 
    {
        std::vector<std::vector<float>> batches;
        for (size_t i = 0; i < all_tokens.size(); i += batch_size) 
        {
            size_t end = std::min(i + batch_size, all_tokens.size());
            std::vector<size_t> batch_tokens(all_tokens.begin() + i, all_tokens.begin() + end);
            batches.push_back(Batch(batch_tokens));
        }
        return batches;
    }

    class Autograd
    {
private:

    };


};

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(tensor_module, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<float>, size_t, size_t>())
        .def("embeddingLookup", &Tensor::embeddingLookup)
        .def("Batch", &Tensor::Batch)
        .def("createBatches", &Tensor::createBatches);
}
