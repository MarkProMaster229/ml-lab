#include <iostream>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
using namespace std;
// базовый токенизатор
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
//я думаю - без класса тензор реализовать многоголовый трасформер будет проблемно 
class Tensor
{
    public:
vector<vector<vector<int>>> tensor = {
    { {0,0}, {0,0} },   // batch 0
    { {0,0}, {0,0} },   // batch 1
    { {0,0}, {0,0} } // batch 2
};


};
//размечаем эмбдинги
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

//далее решаем задачу позиционирования слов для обработки через слой трансформера 
//В трансформере порядок слов важен, потому что сам трансформер не видит порядок.
class PositionalEncoding 
{
    int dim;//это размерность эмбеддинга
    int max_len; // максимальная длина последовательности

    /*
    positions — это двумерный массив, в котором каждая строка соответствует позиции токена,
    а каждый столбец — компоненте позиционного вектора.
    Размерность: [max_len][dim].
    */
    vector<vector<float>> positions;
    public:
    // то что происходит ниже для меня пока магия но оно рабоает - у эмбдинга появляется позиция!
     PositionalEncoding(int dim_, int max_len_ = 512) : dim(dim_), max_len(max_len_) 
     {
        positions.resize(max_len, vector<float>(dim, 0.0f));
            for (int pos = 0; pos < max_len; ++pos) 
            {
                for (int i = 0; i < dim; ++i) 
                {
                    positions[pos][i] = pos / pow(10000.0, 2.0 * (i / 2) / dim);
                    if (i % 2 == 0)
                    positions[pos][i] = sin(positions[pos][i]);
                else
                    positions[pos][i] = cos(positions[pos][i]);

                }
            }
     }

    vector<vector<float>> add_to_embeddings(const vector<vector<float>>& embeddings) 
    {
        vector<vector<float>> out = embeddings;
        for (size_t i = 0; i < embeddings.size(); ++i) 
        {
            for (int j = 0; j < dim; ++j) 
            {
                out[i][j] += positions[i][j];

            }
        }
        return out;
    }
    //магия закончилась 

};

class Transformer 
{
public:
//тут тупо базовые манипуляции с матрицами не более
    // Умножение двух матриц A * B
    vector<vector<float>> matmul(const vector<vector<float>>& A, const vector<vector<float>>& B) {
        size_t n = A.size();           // количество строк в A
        size_t m = A[0].size();        // количество столбцов в A
        size_t p = B[0].size();        // количество столбцов в B
        vector<vector<float>> C(n, vector<float>(p, 0.0f));
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                for (size_t k = 0; k < m; ++k) {
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
    }

    // Транспонирование матрицы
    vector<vector<float>> transpose(const vector<vector<float>>& A) {
        size_t n = A.size();
        size_t m = A[0].size();
        vector<vector<float>> T(m, vector<float>(n, 0.0f));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < m; ++j) {
                T[j][i] = A[i][j];
            }
        }
        return T;
    }
//вот до сих пор 
    // Softmax по строкам
    vector<vector<float>> softmax(const vector<vector<float>>& A) {
        vector<vector<float>> S = A;
        for (size_t i = 0; i < A.size(); ++i) {
            float max_val = *max_element(A[i].begin(), A[i].end());
            float sum = 0.0f;
            for (float val : A[i]) sum += exp(val - max_val);
            for (size_t j = 0; j < A[i].size(); ++j) S[i][j] = exp(A[i][j] - max_val) / sum;
        }
        return S;
    }

};


int main()
{
    cout << "Hello World!\n";
    Tokenizer::ByteTokinizer tok;

    string word = "hello world  ";
    vector<int> encoded = tok.encode(word);
    string decoded = tok.decode(encoded);

    cout << "Input: " << word << endl;

    cout << "Encoded: ";
    for (int id : encoded) cout << id << " ";
    cout << endl;
 // эмбеддинги размерностью 4
    Embedding embedding(128);//эмбдинг и позиционный эмдинг в данном случае имеют прямую зависимость 
    vector<vector<float>> embedded = embedding.encode_to_embedding(word);

    // добавляем позиционное кодирование
    // проясню - первое число на размерность эмбдинга( должен быть такой же как и embedding выше) второе число - максимальная длина последовательности
    //говоря проще - это лимит, сколько токенов трансформер может “увидеть” одновременно.
    PositionalEncoding pe(128, 256);//не забыть про адекватность. И да, эмбдингов это тоже касается
    
    vector<vector<float>> embedded_with_pos = pe.add_to_embeddings(embedded);

    cout << "Decoded: " << decoded << endl;

    cout << "Embeddings:" << endl;
    for (size_t i = 0; i < embedded.size(); i++)
    {
        cout << "Token " << encoded[i] << ": ";
        for (float val : embedded[i])
            cout << val << " ";
        cout << endl;
    }
    cout << "Embeddings positional:" << endl;
    for (size_t i = 0; i < embedded_with_pos.size(); i++)
    {
        cout << "Token " << encoded[i] << ": ";
        for (float val : embedded_with_pos[i])
            cout << val << " ";
        cout << endl;
    }
    

    return 0;
    
}
