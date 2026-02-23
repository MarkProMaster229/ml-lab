#pragma once
#include "optimizer.hpp"

class SGD :public Optimizer
{
    public:
    float lr;//шаг(learning rate)
    SGD(float leaning_rate) : lr(leaning_rate){}

    void step(Tensor& param, const Tensor& grad) override
    {
    // param = param - lr * grad
            for (size_t i = 0; i < param.size(); ++i) {
            param.data[i] -= lr * grad.data[i];
        }
    
    }


};