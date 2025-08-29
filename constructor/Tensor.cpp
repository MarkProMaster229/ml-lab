#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
using namespace std;

class Tensor
{
    public:
    vector<int> data;
    size_t batch_size, seq_len, dim;
    /*
    batch_size — сколько предложений/батчей в тензоре
    seq_len — сколько токенов в каждом предложении
    dim — размерность эмбеддинга (вектор, который хранит каждый токен)
    size_t гарантирует, что все размеры неотрицательные и подходящего типа для адресации памяти.
    */

    Tensor(size_t b, size_t s, size_t d) : batch_size(b), seq_len(s), dim(d) {
        data.resize(b * s * d, 0); // все элементы нули
    }

    int& at(size_t b, size_t s, size_t d) {
        return data[b * seq_len * dim + s * dim + d];
    }

    // метод для вывода части тензора (например, первых n батчей)
    void print(size_t n_batches = 1) {
        for(size_t b = 0; b < n_batches && b < batch_size; ++b){
            cout << "Batch " << b << ":\n";
            for(size_t s = 0; s < seq_len; ++s){
                cout << "  Token " << s << ": ";
                for(size_t d = 0; d < dim; ++d){
                    cout << at(b,s,d) << " ";
                }
                cout << "\n";
            }
            cout << "\n";
        }
    }
};

int main() {
    Tensor t(100, 10, 4); // уменьшил dim, чтобы было проще выводить
    t.at(0,0,0) = 42;     // первый элемент первого токена первого предложения
    t.at(0,0,1) = 1;
    t.at(0,0,2) = 2;
    t.at(0,0,3) = 3;

    // выводим только первый батч
    t.print(1);

    return 0;
}