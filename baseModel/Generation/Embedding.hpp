#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "/mnt/storage/product/ml-lab/baseModel/json.hpp"
#include "Tokenizer.hpp"

using json = nlohmann::json;

class Embedding {
public:
    Tokenizer tokenizator;

    std::vector<std::vector<int>> GetAnaliz() {
        std::ifstream file("test.json");
        if (!file.is_open()) {
            std::cerr << "Cannot open JSON file!" << std::endl;
            return {};
        }

        json j;
        file >> j;
        return processJSON(j);
    }

private:
    std::vector<std::vector<int>> processJSON(const json& j) {
        std::vector<std::vector<int>> sequences;

        for (auto& [key, value] : j.items()) {
            if (value.is_array()) {
                std::string sentence;
                for (auto& w : value) {
                    if (w.is_string())
                        sentence += w.get<std::string>();
                }
                sequences.push_back(tokenizator.myTokinezer(sentence));
            }
        }
        return sequences;
    }
};
