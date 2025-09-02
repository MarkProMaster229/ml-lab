#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <unordered_map>
using namespace std;

class Tokenizer
{
    public:
        static constexpr int PAD = 256;//padding (добивание пустыми местами в батче, когда длины разные).
        static constexpr int BOS = 257;//begin of sequence, токен начала последовательности.
        static constexpr int EOS = 258;//end of sequence, токен конца последовательности.
        static constexpr int UNK = 259;//unknown (неизвестный токен, если что-то не получилось закодировать).
        
        static constexpr int BASE = 256;//просто размер базовой части (все байты).
        static constexpr int SIZE = 260;//общее количество токенов = 256 байтов + 4 спец-токена.
        void myTokinezer(string text)
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
            
            vector<int> token;
            token.push_back(BOS);
            for (char i:text)
            {
                if(mapAllIcon.count(i))
                   token.push_back(mapAllIcon[i]);
                else
                token.push_back(UNK);
            }
            token.push_back(EOS);
            for (int i = 0; i<token.size(); i++)
            {
                cout<<token[i] <<" ";
            }
            cout << endl;
            

        }
    
};
