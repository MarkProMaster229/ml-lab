#include "/mnt/storage/product/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Position.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/MaskGenerator.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Generation/Embedding.hpp"
#include "/mnt/storage/product/ml-lab/baseModel/Model/AttentionLogits.cpp"
// пересмотрена реализация на более мягкую(уже без хард кода))) )

Tensor matmul(const Tensor& A, const Tensor& B_transposed) {
    int batch = A.shape[0];
    int seq_len = A.shape[1];
    int dim = A.shape[2]; // размерность Q/K

    Tensor result(batch, seq_len, seq_len); // внимание квадратное

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < seq_len; ++i) {
            for (int j = 0; j < seq_len; ++j) {
                float sum = 0.0f;
                for (int k = 0; k < dim; ++k) {
                    sum += A.at(b, i, k) * B_transposed.at(b, j, k); // B уже транспонировано
                }
                result.at(b, i, j) = sum;
            }
        }
    }
    return result;
}

Tensor transpose(const Tensor& X) {
    int batch = X.shape[0];
    int seq_len = X.shape[1];
    int dk = X.shape[2];

    Tensor Y(batch, dk, seq_len);

    for (int b = 0; b < batch; ++b)
        for (int i = 0; i < seq_len; ++i)
            for (int j = 0; j < dk; ++j)
                Y.at(b, j, i) = X.at(b, i, j);

    return Y;
}


Tensor softmax(const Tensor& X) {
    Tensor result = X; // создаём копию

    int batch = X.shape[0];
    int rows = X.shape[1];
    int cols = X.shape[2];

    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < rows; ++i) {
            float max_val = -1e30f;
            for (int j = 0; j < cols; ++j) {
                if (X.at(b, i, j) > max_val) max_val = X.at(b, i, j);
            }

            float sum_exp = 0.0f;
            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) = std::exp(X.at(b, i, j) - max_val); // для стабильности
                sum_exp += result.at(b, i, j);
            }

            for (int j = 0; j < cols; ++j) {
                result.at(b, i, j) /= sum_exp;
            }
        }
    }

    return result;
}



class Transformer {
public:
    Transformer(int embedding_dim, int dk);

    void load_weights(const std::string& filename);


private:
    int embedding_dim;
    int dk;
    Logit::AttentionWeights weights;

public:

    Transformer::Transformer(int embedding_dim_, int dk_)
    : embedding_dim(embedding_dim_), dk(dk_), weights() {}
     Tensor forward(Tensor& X) // одна "голова внимания"
     {
    Logit::QKV qkv = Logit().complitle(X, weights);
    Tensor scores = matmul(qkv.Q, transpose(qkv.K)) / sqrt(float(dk));
    Tensor attn_weights = softmax(scores);
    Tensor output = matmul(attn_weights, qkv.V);
    return output;
     }

};