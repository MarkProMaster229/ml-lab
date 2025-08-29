#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
using namespace std;

class Tensor {
public:
    vector<vector<vector<int>>> tensor = {
        { {0,0}, {0,0} },
        { {0,0}, {0,0} },
        { {0,0}, {0,0} }
    };
};

int main() {
    Tensor t;
    cout << t.tensor[0][0][0] << endl;
    return 0;
}