#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "Tokenizer.cpp"
#include "Tensor.cpp"

using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");
    Tokenizer tok;
    string a = "привет тебе !";
    tok.myTokinezer(a);


    Tensor t(2, 3, 10);

    // Заполняем случайными числами
    for (int i = 0; i < t.shape[0]; ++i)
        for (int j = 0; j < t.shape[1]; ++j)
            for (int k = 0; k < t.shape[2]; ++k)
                t.at(i, j, k) = static_cast<float>(i + j + k);

    t.save("tensor.pt");  // сохраняем на диск

    Tensor t2;
    t2.load("tensor.pt"); // загружаем обратно

    cout << "t2[1][2][3] = " << t2.at(1, 2, 3) << endl;

}