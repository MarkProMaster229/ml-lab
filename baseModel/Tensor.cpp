#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "json.hpp"



using json = nlohmann::json;

using namespace std;

class Tensor
{
public:
    vector<float> data;    // хранит все числа
    vector<int> shape;     // размеры по осям, [batch, seq_len, embedding_dim]
    vector<int> strides;   // смещения по каждой оси

    // Конструктор с заданными размерами
    Tensor(int batch, int seq_len, int embedding_dim) {
        shape = {batch, seq_len, embedding_dim};
        data.resize(batch * seq_len * embedding_dim);

        strides.resize(3);
        strides[2] = 1;
        strides[1] = embedding_dim;
        strides[0] = seq_len * embedding_dim;
    }

    // Конструктор пустого тензора (для загрузки)
    Tensor() {}

    // Доступ к элементу [i,j,k]
    float& at(int i, int j, int k) {
        assert(i >= 0 && i < shape[0]);
        assert(j >= 0 && j < shape[1]);
        assert(k >= 0 && k < shape[2]);
        int index = i*strides[0] + j*strides[1] + k*strides[2];
        return data[index];
    }

    // Сохранение тензора в бинарный файл
    void save(const string& filename) {
        ofstream file(filename, ios::binary);
        if (!file) {
            cerr << "Error opening file for writing: " << filename << endl;
            return;
        }

        int dims = shape.size();
        file.write(reinterpret_cast<char*>(&dims), sizeof(int));
        file.write(reinterpret_cast<char*>(shape.data()), dims * sizeof(int));

        int total = data.size();
        file.write(reinterpret_cast<char*>(data.data()), total * sizeof(float));

        file.close();
    }

    // Загрузка тензора из бинарного файла
    void load(const string& filename) {
        ifstream file(filename, ios::binary);
        if (!file) {
            cerr << "Error opening file for reading: " << filename << endl;
            return;
        }

        int dims;
        file.read(reinterpret_cast<char*>(&dims), sizeof(int));
        shape.resize(dims);
        file.read(reinterpret_cast<char*>(shape.data()), dims * sizeof(int));

        // пересчёт strides
        strides.resize(dims);
        strides[dims - 1] = 1;
        for (int i = dims - 2; i >= 0; --i) {
            strides[i] = shape[i + 1] * strides[i + 1];
        }

        int total = 1;
        for (int s : shape) total *= s;
        data.resize(total);
        file.read(reinterpret_cast<char*>(data.data()), total * sizeof(float));

        file.close();
    }
};

class Embeding {
public:
    Tokenizer tokenizator;

    // Метод для обработки JSON и получения токенов
    vector<vector<int>> GetAnaliz() {
        ifstream file("test.json");
        if (!file.is_open()) {
            cerr << "Cannot open JSON file!" << endl;
            return {};
        }

        json j;
        file >> j;

        return processJSON(j);
    }

private:
    // Вспомогательный метод, который превращает JSON в массив токенов
    vector<vector<int>> processJSON(const json& j) {
        vector<vector<int>> sequences;

        for (auto& [key, value] : j.items()) {
            if (value.is_array()) {
                string sentence;
                for (auto& w : value) {
                    if (w.is_string()) sentence += w.get<string>();
                }
                sequences.push_back(tokenizator.myTokinezer(sentence));
            }
        }

        return sequences;
    }
};
