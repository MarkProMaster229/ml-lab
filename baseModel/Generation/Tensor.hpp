#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <cassert>
#include <iostream>

class Tensor {
public:
    std::vector<float> data;    // хранит числа тензора
    std::vector<int> shape;     // размеры [batch, seq_len, embedding_dim]
    std::vector<int> strides;   // смещения по осям

    // Конструктор пустого тензора (для загрузки)
    Tensor() {}

    // Конструктор с размерами (создает пустой тензор)
    Tensor(int batch, int seq_len, int embedding_dim) {
        shape = {batch, seq_len, embedding_dim};
        data.resize(batch * seq_len * embedding_dim);

        strides.resize(3);
        strides[2] = 1;
        strides[1] = embedding_dim;
        strides[0] = seq_len * embedding_dim;
    }

    // Обычный доступ к элементу (для изменения)
    float& at(int i, int j, int k) {
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        int index = i * strides[0] + j * strides[1] + k * strides[2];
        return data[index];
    }

    // Константный доступ к элементу (только для чтения)
    const float& at(int i, int j, int k) const {
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        int index = i * strides[0] + j * strides[1] + k * strides[2];
        return data[index];
    }

    // Сохраняем тензор в файл
    void save(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file for writing: " << filename << std::endl;
            return;
        }

        int dims = shape.size();
        file.write(reinterpret_cast<const char*>(&dims), sizeof(int));
        file.write(reinterpret_cast<const char*>(shape.data()), dims * sizeof(int));

        int total = data.size();
        file.write(reinterpret_cast<const char*>(data.data()), total * sizeof(float));
        file.close();
    }

    // Загружаем тензор из файла
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            std::cerr << "Cannot open file for reading: " << filename << std::endl;
            return;
        }

        int dims;
        file.read(reinterpret_cast<char*>(&dims), sizeof(int));
        shape.resize(dims);
        file.read(reinterpret_cast<char*>(shape.data()), dims * sizeof(int));

        // пересчёт strides
        strides.resize(dims);
        strides[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i)
            strides[i] = shape[i + 1] * strides[i + 1];

        int total = 1;
        for (int s : shape) total *= s;
        data.resize(total);
        file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));
        file.close();
    }
};
