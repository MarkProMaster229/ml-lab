#include <iostream>
#include <vector>
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/Generation/Tensor.hpp"
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/Model/Transformer.hpp"
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/backprop/CrossEntropyLoss.hpp"
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/backprop/optimizer.hpp"


Tensor softmax(const Tensor& transformer_out, const std::vector<float>& W_out, const std::vector<float>& b_out) {
    int batch_size = transformer_out.shape[0];
    int seq_len = transformer_out.shape[1];
    int dk = transformer_out.shape[2];
    int vocab_size = b_out.size();

    Tensor out_probs(batch_size, seq_len, vocab_size);

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < seq_len; j++) {
            std::vector<float> logits(vocab_size, 0.0f);
            for (int v = 0; v < vocab_size; v++) {
                for (int k = 0; k < dk; k++)
                    logits[v] += transformer_out.at(i, j, k) * W_out[v * dk + k];
                logits[v] += b_out[v];
            }

            // softmax
            float max_logit = *std::max_element(logits.begin(), logits.end());
            float sum_exp = 0.0f;
            for (auto& x : logits) sum_exp += (x = std::exp(x - max_logit));
            for (int v = 0; v < vocab_size; v++) logits[v] /= sum_exp;

            for (int v = 0; v < vocab_size; v++)
                out_probs.at(i, j, v) = logits[v];
        }
    }

    return out_probs;
}
int finalcut()
{
    int epochs = 10;
    int batch_size = 2;
    int seq_len = 5;
    int embedding_dim = 128;
    int dk = 64;
    int vocab_size = 50;
    float learning_rate = 0.01f; 


    
}