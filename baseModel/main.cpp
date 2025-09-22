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

    //один раз запускается инициализация матриц весов внимания Wq Wk Wv по этому вынесу запуск сюда чтоб не забыть

    //WeightGenerator gen(128, 128);
    //gen.initialize();
    //gen.save("weights.pt");

    // по хорошу инициализацию тензора тоде сююда вынести для того чтоб он единоразово генерировал эмбединги, а далее только работаем с ними



    return 0;
}
