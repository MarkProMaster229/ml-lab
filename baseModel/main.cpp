#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "Tokenizer.cpp"
#include "Tensor.cpp"
#include "initialization.cpp"
using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");

    Embeding em;
    vector<vector<int>> all_tokens = em.GetAnaliz(); // Получили токены из JSON

    initialization init;
    int embedding_dim = 10;

    // Создаём тензор случайных эмбеддингов для первого предложения
    Tensor t = init.freeRandom(all_tokens[0], embedding_dim);

    t.save("tensor.pt");   // Сохраняем
    Tensor t2;
    t2.load("tensor.pt");  // Загружаем обратно

    cout << "t2[0][0][0] = " << t2.at(0, 0, 0) << endl;

    return 0;
}
