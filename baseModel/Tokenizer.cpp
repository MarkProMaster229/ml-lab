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
        vector<int> myTokinezer(string text)
        {
            unordered_map<char, int> mapAllIconChar;
            unordered_map<string, int> mapAllIconStr;
            vector<int> finalyWorld;
            int rez = 7;

            // Спец-токены (строки)
            mapAllIconStr["[PAD]"] = PAD;
            mapAllIconStr["[BOS]"] = BOS;
            mapAllIconStr["[EOS]"] = EOS;
            mapAllIconStr["[UNK]"] = UNK;
            mapAllIconStr["[BASE]"] = BASE;
            mapAllIconStr["[SIZE]"] = SIZE;

            // Все символы
            for (int i = 0; i < 256; ++i)
                mapAllIconChar[(char)i] = i;

            vector<int> token;
            token.push_back(BOS);
            for (char i:text)
            {
                if(mapAllIconChar.count(i))
                   token.push_back(mapAllIconChar[i]);
                else
                   token.push_back(UNK);
            }
            token.push_back(EOS);
            for (int i = 0; i<token.size(); i++)
            {
                cout<<token[i] <<" ";
            }
            cout << endl;
            return token;
        }

};