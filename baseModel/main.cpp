#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include "initialization.cpp"
using namespace std;

int main()
{
    setlocale(LC_ALL, "Russian");

    initialization init;
    init.run();

    return 0;
}
