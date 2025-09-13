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
    Tokenizer tok;
    string a = "привет тебе !";
    vector token = tok.myTokinezer(a);


    initialization init;
    Tensor t = init.freeRandom(token, 10);

    t.save("tensor.pt");  // сохраняем на диск

    Tensor t2;
    t2.load("tensor.pt"); // загружаем обратно



    Embeding em;
    em.GetAnaliz();

}