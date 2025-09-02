#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "Tokenizer.cpp"

using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");
    Tokenizer tok;
    string a = "привет тебе !";
    tok.myTokinezer(a);

}