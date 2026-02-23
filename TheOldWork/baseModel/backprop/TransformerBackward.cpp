#include "TransformerBackward.hpp"
#include <cmath>
#include <iostream>

#include "TransformerBackward.hpp"
#include <iostream>
#include <cmath>

//шаблон backward для одной головы внимания
Tensor transformer_backward(Transformer& transformer, const Tensor& d_output) {
    // На данный момент будем просто возвращать тот же размер тензора, заполненный нулями
    Tensor dX = d_output; // для проверки размерностей
    for (auto& v : dX.data) v = 0.0f;

    // TODO:
    // 1) Использовать last_X, last_qkv, last_attn_weights (надо их сохранить в Transformer)
    // 2) Вычислить градиенты по формуле attention:
    //    dV = d_output * attn_weights^T
    //    dAttn = d_output * V^T
    //    dScores = dAttn * softmax_grad(scores)
    //    dQ = dScores * K
    //    dK = dScores^T * Q
    //    dX = dQ * Wq^T + dK * Wk^T + dV * Wv^T
    // 3) Вернуть dX
    return dX;
}
