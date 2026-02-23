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
//вот такое решение есть для последовательного хранения offset = b * seq_len * dim + s * dim + d;
class Tensor
{
public:
    // храним все данные в одномерном массиве, теперь тип float для эмбеддингов
    vector<float> data;

    // размеры тензора
    size_t batch_size; // сколько предложений/батчей в тензоре
    size_t seq_len;    // сколько токенов в каждом предложении
    size_t dim;        // размерность эмбеддинга (вектор, который хранит каждый токен)
    /*
    size_t гарантирует, что все размеры неотрицательные и подходящего типа для адресации памяти.
    */

    // конструктор: задаём размеры, инициализируем все элементы нулями
    Tensor(size_t b, size_t s, size_t d) : batch_size(b), seq_len(s), dim(d) {
        data.resize(b * s * d, 0.0f); // все элементы нули
    }

    // доступ к элементу по координатам (b, s, d)
    float& at(size_t b, size_t s, size_t d) {
        return data[b * seq_len * dim + s * dim + d];
    }

    // константная версия доступа для случаев, когда не хотим менять данные
    const float& at(size_t b, size_t s, size_t d) const {
        return data[b * seq_len * dim + s * dim + d];
    }

    // метод для вывода части тензора (например, первых n батчей)
    void print(size_t n_batches = 1) {
        for(size_t b = 0; b < n_batches && b < batch_size; ++b){
            cout << "Batch " << b << ":\n";
            for(size_t s = 0; s < seq_len; ++s){
                cout << "  Token " << s << ": ";
                for(size_t d = 0; d < dim; ++d){
                    cout << at(b,s,d) << " "; // вывод значений по координатам
                }
                cout << "\n";
            }
            cout << "\n";
        }
    }
};


//размечаем эмбдинги
class Embedding : public Tokenizer
{
public:
    int dim; // размерность эмбеддинга, т.е. сколько чисел будет представлять один токен

    /*
    weights[token_id] — это вектор размером dim для конкретного токена.
    Теперь используем Tensor вместо vector<vector<float>>, чтобы хранить эмбеддинги всех токенов.
    */
    Tensor weights;

    // конструктор: инициализация весов нулями
    Embedding(int dim_) : dim(dim_), weights(Vocab::SIZE, 1, dim_) 
    {
        // по сути нули — неидеально для обучения, лучше использовать случайное распределение
        // например: weights.at(token_id, 0, d) = (rand() / (float)RAND_MAX - 0.5f) * 0.01f;
    }

    // Получение эмбеддинга конкретного токена
    // Раньше возвращали vector<float>, теперь просто копируем в Tensor на 1 токен
    Tensor get_embedding(int token_id)
    {
        Tensor emb(1, 1, dim); // 1 батч, 1 токен
        for (int d = 0; d < dim; ++d)
            emb.at(0, 0, d) = weights.at(token_id, 0, d);
        return emb;
    }

    // Кодируем строку в последовательность эмбеддингов
    Tensor encode_to_embedding(const string& text)
    {
        vector<int> token_ids = ByteTokinizer().encode(text);
        Tensor out(1, token_ids.size(), dim); // 1 батч, seq_len = количество токенов

        for (size_t s = 0; s < token_ids.size(); ++s)
        {
            int id = token_ids[s];
            for (int d = 0; d < dim; ++d)
                out.at(0, s, d) = weights.at(id, 0, d);
        }

        return out;
    }
};


// далее решаем задачу позиционирования слов для обработки через слой трансформера 
// В трансформере порядок слов важен, потому что сам трансформер не видит порядок.
class PositionalEncoding 
{
    int dim;      // это размерность эмбеддинга
    int max_len;  // максимальная длина последовательности

    /*
    positions — теперь используем Tensor вместо vector<vector<float>>.
    Каждая строка соответствует позиции токена,
    каждый столбец — компоненте позиционного вектора.
    Размерность: [max_len, 1, dim], где batch_size=1.
    */
    Tensor positions;

public:
    // Конструктор создаёт позиции и заполняет их синусами и косинусами
    // то что происходит ниже для меня пока магия но оно работает — у эмбдинга появляется позиция!
    PositionalEncoding(int dim_, int max_len_ = 512) 
        : dim(dim_), max_len(max_len_), positions(max_len_, 1, dim_)
    {
        for (int pos = 0; pos < max_len; ++pos) 
        {
            for (int i = 0; i < dim; ++i) 
            {
                float angle = pos / pow(10000.0f, 2.0f * (i / 2) / dim);
                if (i % 2 == 0)
                    positions.at(pos, 0, i) = sin(angle);
                else
                    positions.at(pos, 0, i) = cos(angle);
            }
        }
    }

    // Метод добавляет позиционные кодировки к эмбеддингам
    // embeddings — Tensor размерности [1, seq_len, dim]
    Tensor add_to_embeddings(const Tensor& embeddings) 
    {
        Tensor out(1, embeddings.seq_len, embeddings.dim); // создаём новый Tensor для выхода
        for (size_t s = 0; s < embeddings.seq_len; ++s) 
        {
            for (int d = 0; d < dim; ++d) 
            {
                // добавляем позицию к эмбеддингу токена
                out.at(0, s, d) = embeddings.at(0, s, d) + positions.at(s, 0, d);
            }
        }
        return out;
    }
    // магия закончилась
};

class Transformer 
{
public:
    // тут тупо базовые манипуляции с матрицами, не более

    // Умножение двух матриц A * B
    // A: [n, m], B: [m, p] → возвращает Tensor [1, n, p] (batch_size = 1)
    Tensor matmul(const Tensor& A, const Tensor& B) {
        size_t n = A.seq_len;      // количество строк в A
        size_t m = A.dim;          // количество столбцов в A
        size_t p = B.dim;          // количество столбцов в B

        Tensor C(1, n, p); // создаём Tensor для результата

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < p; ++j) {
                float sum = 0.0f;
                for (size_t k = 0; k < m; ++k) {
                    sum += A.at(0, i, k) * B.at(0, k, j);
                }
                C.at(0, i, j) = sum;
            }
        }
        return C;
    }

    // Транспонирование матрицы
    // A: [1, n, m] → возвращает Tensor [1, m, n]
    Tensor transpose(const Tensor& A) {
        Tensor T(1, A.dim, A.seq_len); // меняем местами seq_len и dim
        for (size_t i = 0; i < A.seq_len; ++i) {
            for (size_t j = 0; j < A.dim; ++j) {
                T.at(0, j, i) = A.at(0, i, j);
            }
        }
        return T;
    }

    // Softmax по строкам
    // A: [1, n, m] → возвращает Tensor того же размера
    Tensor softmax(const Tensor& A) {
        Tensor S(1, A.seq_len, A.dim);
        for (size_t i = 0; i < A.seq_len; ++i) {
            float max_val = A.at(0, i, 0);
            for (size_t j = 0; j < A.dim; ++j) {
                if (A.at(0, i, j) > max_val) max_val = A.at(0, i, j);
            }

            float sum = 0.0f;
            for (size_t j = 0; j < A.dim; ++j) {
                sum += exp(A.at(0, i, j) - max_val);
            }

            for (size_t j = 0; j < A.dim; ++j) {
                S.at(0, i, j) = exp(A.at(0, i, j) - max_val) / sum;
            }
        }
        return S;
    }

};


int main()
{
    cout << "Hello World!\n";

    // Создаём базовый токенизатор
    Tokenizer::ByteTokinizer tok;

    string word = "hello world  ";

    // Кодируем строку в последовательность токенов
    vector<int> encoded = tok.encode(word);

    // Декодируем обратно для проверки
    string decoded = tok.decode(encoded);

    cout << "Input: " << word << endl;

    cout << "Encoded: ";
    for (int id : encoded) cout << id << " ";
    cout << endl;

    // эмбеддинги размерностью 128
    // эмбдинг и позиционный эмбеддинг в данном случае имеют прямую зависимость
    Embedding embedding(128);

    // получаем Tensor с эмбеддингами [1, seq_len, dim]
    Tensor embedded = embedding.encode_to_embedding(word);

    // добавляем позиционное кодирование
    // первое число — размерность эмбединга (должен совпадать с embedding выше)
    // второе число — максимальная длина последовательности
    // проще говоря — это лимит, сколько токенов трансформер может “увидеть” одновременно
    PositionalEncoding pe(128, 256);

    Tensor embedded_with_pos = pe.add_to_embeddings(embedded);

    cout << "Decoded: " << decoded << endl;

    cout << "Embeddings:" << endl;
    for (size_t s = 0; s < embedded.seq_len; ++s)
    {
        cout << "Token " << encoded[s] << ": ";
        for (int d = 0; d < embedded.dim; ++d)
            cout << embedded.at(0, s, d) << " ";
        cout << endl;
    }

    cout << "Embeddings positional:" << endl;
    for (size_t s = 0; s < embedded_with_pos.seq_len; ++s)
    {
        cout << "Token " << encoded[s] << ": ";
        for (int d = 0; d < embedded_with_pos.dim; ++d)
            cout << embedded_with_pos.at(0, s, d) << " ";
        cout << endl;
    }

    return 0;
}
