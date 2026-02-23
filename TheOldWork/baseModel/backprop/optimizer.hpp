#pragma once 
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/Generation/Tensor.hpp"

class Optimizer
{
    public:
    virtual ~Optimizer() = default;
    virtual void step(Tensor& param, const Tensor& grad) = 0;
    //сбрасывать состояния, если нужны моменты
    virtual void zero_grad() {}
};