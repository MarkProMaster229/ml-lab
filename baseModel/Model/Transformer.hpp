#include "/mnt/storage/product/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Position.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"


class Transformer {
public:
    struct QKV {
        Tensor Q;
        Tensor K;
        Tensor V;
    };

    Transformer(int embedding_dim, int dk);

    void load_weights(const std::string& filename);
    QKV head(Tensor& X);

private:
    int embedding_dim;
    int dk;
    std::vector<float> Wq, Wk, Wv;
};
