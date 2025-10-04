#include <iostream>
#include <cmath>
#include "Transformer.cpp"


// нужна матрица весов для линейного слоя после трасформера
/*
она проецирует выходной тензор трасформера размером - dk
в размер словаря(vocab_size)
W_out: vocab_size * dk

b_out — это смещение (bias) для линейного слоя.
b_out: vocab_size
Он добавляется к результату матричного умножения, чтобы сдвинуть логиты.


*/


    //std::vector<float> predict_next_token_probs(const Tensor& transformer_output, int batch_index) {
    //Tensor logits = linear(transformer_output, W_out, b_out);
    //Tensor probs = softmax(logits);
    //return probs[batch_index][transformer_output.shape[1]-1];
//}

//int predict_next_token(const Tensor& transformer_output, int batch_index) {
    //auto probs = predict_next_token_probs(transformer_output, batch_index);
    //return std::distance(probs.begin(), std::max_element(probs.begin(), probs.end()));
//}