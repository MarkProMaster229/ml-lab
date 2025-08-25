#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
using namespace std;

class Tokenizer
{
    public:

    //словарь как ASCII и в навесок резервируем места старше 255(конец нумерации последнего символа)
     struct Vocab
     {
        /*
        Вокаб даёт нам правила соответствия:
        если встретили спец-токен → добавляем его id (например, EOS = 258).
        если встретили символ → берём его байтовое значение (например, 'h' = 104).
        
        Таким образом токенизатор не придумывает id случайно, он просто смотрит в таблицу «символ → число».
        */
        static constexpr int PAD = 256;//padding (добивание пустыми местами в батче, когда длины разные).
        static constexpr int BOS = 257;//begin of sequence, токен начала последовательности.
        static constexpr int EOS = 258;//end of sequence, токен конца последовательности.
        static constexpr int UNK = 259;//unknown (неизвестный токен, если что-то не получилось закодировать).
        
        static constexpr int BASE = 256;//просто размер базовой части (все байты).
        static constexpr int SIZE = 260;//общее количество токенов = 256 байтов + 4 спец-токена.
     };
     // основа токенизатора 
     struct ByteTokinizer
     {
        vector<int> encode(const string& text, bool add_bos=true, bool add_eos=true)
        {
            //Создаём пустой список токенов (ids)
            //Если нужно — сразу добавляем спец-токен начала (BOS = 257).
            vector<int> ids;
            if(add_bos) ids.push_back(Vocab::BOS);
            
            /*
            Берём строку посимвольно.
            Каждый символ в C++ хранится в виде байта (число 0–255).
            Кладём это число в список токенов.
            Например "hi" - 'h' = 104, 'i' = 105.
            */
            for(unsigned char c: text)
            {
                ids.push_back((int)c);// каждый байт напрямую в id
            }
            //брат если слово кончилось кинь токен конца последовательности епта
            //Возвращаем готовый список.
            //Итого "hi" → [257, 104, 105, 258].
            //где 257 - сообщает что это начало, 258 - конец, 104 -"h", 105 - "i" 
            if (add_eos) ids.push_back(Vocab::EOS);
            
            return ids;
            
        }
        //декодер для примера и проверки работоспособности самого токенизатора
        string decode(const vector<int>& ids) 
        {
        string out;
        for (int id : ids) 
        {
            if (0 <= id && id < Vocab::BASE) 
            {
                out.push_back((char)id); // байт → символ
            }
            // спец-токены (BOS, EOS, PAD) обычно пропускаем, не нужны нам заастую, но могут пригодится для анализа и поска ошибок в
            // обучающем датасете
        }
        return out;
    }
    
};

};

class Embedding : public Tokenizer
{
    public:
    int dim;//это размерность эмбеддинга, т.е. сколько чисел будет представлять один токен.
    /*
    weights[token_id] — это вектор размером dim для конкретного токена.
    vector<vector<float>> — двумерный вектор: строки — это токены, столбцы — параметры/веса (эмбеддинги).
    */
    vector<vector<float>> weights;
    //можно менять размер матрицы весов ради прикола можно поставить 512 но не рекомендую
    Embedding(int dim_) : dim(dim_) 
    {
        weights.resize(Vocab::SIZE, vector<float>(dim, 0.0f));// инициализация нулями - но по сути это не совсем верно 
        //обьясняю почему - при обучении намного легче будет корректирваоть ошибку методом обратного распространения 
        //если веса будут изначально разбросаны, в ином случае все наши слова равны по смыслу с начала

    }
    //Получение эмбеддинга конкретного токена
    //Взял токен по его id и вернул вектор эмбеддинга этого токена.
    //Пример: токен 'h' = 104 → вернётся weights[104], который размерностью dim.
    vector<float> get_embedding(int token_id)
    {
        return weights[token_id];
    }
    /*
    Сначала мы токенизируем текст через ByteTokinizer(), получаем список токенов.
    Потом для каждого токена берём его эмбеддинг через get_embedding.
    В итоге получаем матрицу эмбеддингов размерности [кол-во токенов, dim].
    */
    vector<vector<float>> encode_to_embedding(const string& text)
    {
        vector<int> token_ids = ByteTokinizer().encode(text);
        vector<vector<float>>out;
        for(int id : token_ids) out.push_back(get_embedding(id));
        return out;
    }

};

int main()
{
    cout << "Hello World!\n";
    Tokenizer::ByteTokinizer tok;

    string word = "привет как твои дела ";
    vector<int> encoded = tok.encode(word);
    string decoded = tok.decode(encoded);

    cout << "Input: " << word << endl;

    cout << "Encoded: ";
    for (int id : encoded) cout << id << " ";
    cout << endl;

    Embedding embedding(4);
    vector<vector<float>> embedded = embedding.encode_to_embedding(word);

    cout << "Decoded: " << decoded << endl;
    cout << "Embeddings:" << endl;
    for (size_t i = 0; i < embedded.size(); i++)
    {
        cout << "Token " << encoded[i] << ": ";
        for (float val : embedded[i])
            cout << val << " ";
        cout << endl;
    }

    return 0;
}
