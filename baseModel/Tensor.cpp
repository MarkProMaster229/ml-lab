#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <assert.h>
using namespace std;

class Tensor
{
    public:
    vector<float> data;// хранит все числа
    vector<int> shape;//размеры по осям, [batch, seq_len, embedding_dim]
    vector<int> strides;// смещения по каждой оси

     Tensor(int batch, int seq_len, int embedding_dim) {
        shape = {batch, seq_len, embedding_dim};
        // выделяем память под все элементы
        data.resize(batch * seq_len * embedding_dim);

        // считаем strides
        strides.resize(3);
        strides[2] = 1;  // последняя ось
        strides[1] = embedding_dim;
        strides[0] = seq_len * embedding_dim;
    }

        // доступ к элементу [i,j,k]
    float& at(int i, int j, int k) {
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        int index = i*strides[0] + j*strides[1] + k*strides[2];
        return data[index];
    }

};