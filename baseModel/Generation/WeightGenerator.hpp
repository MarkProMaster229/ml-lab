#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <random>

class WeightGenerator {
public:
    // Конструктор: задаём размеры embedding_dim и dk
    WeightGenerator(int embedding_dim, int dk)
        : embedding_dim(embedding_dim), dk(dk),
          Wq(embedding_dim * dk),
          Wk(embedding_dim * dk),
          Wv(embedding_dim * dk) {}

    // Генерация случайных весов (Xavier)
    void initialize() {
        float stddev = std::sqrt(2.0f / (embedding_dim + dk));
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0f, stddev);

        for (auto &w : Wq) w = dist(gen);
        for (auto &w : Wk) w = dist(gen);
        for (auto &w : Wv) w = dist(gen);
    }

    // Сохраняем в PT (бинарный)
    void save(const std::string &filename) {
        std::ofstream out(filename, std::ios::binary);
        out.write(reinterpret_cast<char*>(Wq.data()), Wq.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(Wk.data()), Wk.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(Wv.data()), Wv.size() * sizeof(float));
        out.close();
    }

    // Загружаем из PT
    void load(const std::string &filename) {
        std::ifstream in(filename, std::ios::binary);
        in.read(reinterpret_cast<char*>(Wq.data()), Wq.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(Wk.data()), Wk.size() * sizeof(float));
        in.read(reinterpret_cast<char*>(Wv.data()), Wv.size() * sizeof(float));
        in.close();
    }

    // Геттеры для Wq/Wk/Wv
    std::vector<float>& getWq() { return Wq; }
    std::vector<float>& getWk() { return Wk; }
    std::vector<float>& getWv() { return Wv; }

private:
    int embedding_dim;
    int dk;
    std::vector<float> Wq, Wk, Wv;
};
