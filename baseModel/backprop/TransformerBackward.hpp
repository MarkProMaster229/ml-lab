#pragma once
#include "/home/chelovek/Загрузки/mllub/ml-lab/baseModel/Model/Transformer.hpp"

Tensor transformer_backward(Transformer& transformer, const Tensor& d_output);
