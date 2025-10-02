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
    // так надо
        Tensor operator/(float scalar) const {
        Tensor result = *this; // создаём копию
        for (auto& v : result.data)
            v /= scalar;
        return result;
    }

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
    /*
Я на лошади. Ты на белом коне, и я на белом коне. А потом на бал. А потом салют… в нашу честь. Салют в нашу
честь. Сначала на парад, а потом в дом офицеров пойдём. Бал будет. Пиво. Салют.
В мою честь салют. Я полковник и ты — на белом коне. Ты меня слышишь?
Посмотри, какая у меня форма! Посмотри, какие у неё звезды! Я её ещё никому не показывал. Только тебе покажу. Посмотри!
Это моя парадная. Специально сшил! Посмотри, не бойся! Погоны? Я их поменяю!
Вот они, погоны. Вот! Настоящие! Видишь? Полковник. Я полковник. Я их на парад надену. Это моя форма.
Самая чистая. Что ты кашляешь? Вот пойдём в дом офицеров, пива выпьем — и кашлять не будешь.
Я с кортиком, на белом коне, командую парадом! Я полковник! Я командую парадом! Я в звёздах! На белом коне!
Я полковник! Я командую парадом!
    */

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
