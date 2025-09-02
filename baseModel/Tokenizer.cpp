#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
using namespace std;

class Tokenizer
{
    public:
        static constexpr int PAD = 1;//padding (добивание пустыми местами в батче, когда длины разные).
        static constexpr int BOS = 2;//begin of sequence, токен начала последовательности.
        static constexpr int EOS = 3;//end of sequence, токен конца последовательности.
        static constexpr int UNK = 4;//unknown (неизвестный токен, если что-то не получилось закодировать).
        
        static constexpr int BASE = 5;//просто размер базовой части (все байты).
        static constexpr int SIZE = 6;//общее количество токенов = 256 байтов + 4 спец-токена.

        void myTokinezer(string a)
        {
            unordered_map<char, int> mapAllIcon;
            vector<int> finalyWorld;
            int rez = 7;
            mapAllIcon['[PAD]'] = PAD;
            mapAllIcon['[BOS]'] = BOS;
            mapAllIcon['[EOS]'] = EOS;
            mapAllIcon['[UNK]'] = UNK;
            mapAllIcon['[BASE]'] = BASE;
            mapAllIcon['[SIZE]'] = SIZE;

            string rus = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя";
            for (char i : rus)
            {
                mapAllIcon[i] = rez++;
            }

            finalyWorld[0] = 2;

            

            

            cout<<"test";

            for(int i = 0;i<44;i++)
            {
                cout<<mapAllIcon[i];
            }
        }
    
};
