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

    // Сохраняем в PT (бинарный) с заголовком
    void save(const std::string &filename) {
        std::ofstream out(filename, std::ios::binary);
        if (!out) {
            std::cerr << "Не удалось открыть файл для записи: " << filename << std::endl;
            return;
        }

        // Записываем заголовок (embedding_dim и dk)
        out.write(reinterpret_cast<char*>(&embedding_dim), sizeof(int));
        out.write(reinterpret_cast<char*>(&dk), sizeof(int));

        // Записываем сами веса
        out.write(reinterpret_cast<char*>(Wq.data()), Wq.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(Wk.data()), Wk.size() * sizeof(float));
        out.write(reinterpret_cast<char*>(Wv.data()), Wv.size() * sizeof(float));

        out.close();
    }

    // Загружаем из PT с учётом заголовка
    void load(const std::string &filename) {
        std::ifstream in(filename, std::ios::binary);
        if (!in) {
            std::cerr << "Не удалось открыть файл для чтения: " << filename << std::endl;
            return;
        }

        // Читаем заголовок
        int file_embedding_dim = 0;
        int file_dk = 0;
        in.read(reinterpret_cast<char*>(&file_embedding_dim), sizeof(int));
        in.read(reinterpret_cast<char*>(&file_dk), sizeof(int));

        // Проверяем соответствие размеров
        if (file_embedding_dim != embedding_dim || file_dk != dk) {
            std::cerr << "Размерности в файле не совпадают с текущими настройками." << std::endl;
            std::cerr << "Файл: embedding_dim=" << file_embedding_dim << ", dk=" << file_dk
                      << " | Текущие: embedding_dim=" << embedding_dim << ", dk=" << dk << std::endl;
            in.close();
            return;
        }

        // Считываем веса
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
