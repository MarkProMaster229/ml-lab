import numpy as np
import os

# Настройка вывода NumPy — печатать все элементы
np.set_printoptions(threshold=100)

# --- Работа с tensor.pt - он ручной разумеется без совместимости с pickle ---
tensor_file = "/mnt/storage/product/ml-lab/baseModel/tensor.pt"

try:
    with open(tensor_file, "rb") as f:
        dims = np.frombuffer(f.read(4), dtype=np.int32)[0]  # количество осей
        shape = np.frombuffer(f.read(dims * 4), dtype=np.int32)
        total = np.prod(shape)
        data_tensor = np.frombuffer(f.read(total * 4), dtype=np.float32)
    print(f"[INFO] tensor.pt shape: {shape}")
    print(f"[INFO] tensor.pt data (первые 10 элементов): {data_tensor[:10]}")
except Exception as e:
    print(f"[ERROR] Ошибка при чтении tensor.pt: {e}")

# --- Работа с токенами ---
BOS = 257
EOS = 258
tokens = [
    257, 104, 101, 108, 108, 111, 119, 111, 114, 108, 100, 116, 101, 115, 116, 258,
    257, 208, 191, 209, 128, 208, 184, 208, 178, 208, 181, 209, 130, 32, 209, 130, 208, 181, 208, 177, 208, 181, 32, 208, 180, 209, 128, 209, 131, 208, 179, 258,
    257, 208, 186, 208, 176, 208, 186, 32, 208, 180, 208, 181, 208, 187, 208, 176, 32, 209, 130, 208, 178, 208, 190, 208, 184, 32, 258
]

byte_vals = [t for t in tokens if t != BOS and t != EOS]
try:
    decoded = bytes(byte_vals).decode('utf-8')
    print(f"[INFO] Декодированные токены: {decoded}")
except Exception as e:
    print(f"[ERROR] Ошибка при декодировании токенов: {e}")

# --- Работа с weights.pt с заголовком ---
weights_file = "/mnt/storage/product/ml-lab/baseModel/weights.pt"

if not os.path.exists(weights_file):
    print(f"[ERROR] Файл {weights_file} не найден!")
else:
    try:
        with open(weights_file, "rb") as f:
            # --- Читаем заголовок ---
            embedding_dim = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            dk = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
            print(f"[INFO] Заголовок файла: embedding_dim={embedding_dim}, dk={dk}")

            # --- Читаем все веса ---
            data = np.frombuffer(f.read(), dtype=np.float32)
            num_elements = data.size
            print(f"[INFO] Количество элементов после заголовка: {num_elements}")

            size_per_matrix = embedding_dim * dk
            if num_elements != size_per_matrix * 3:
                print(f"[WARNING] Ожидалось {size_per_matrix*3} элементов, а найдено {num_elements}. Возможно, файл повреждён.")

            # --- Разделяем на Wq, Wk, Wv ---
            Wq = data[0:size_per_matrix].reshape(embedding_dim, dk)
            Wk = data[size_per_matrix:2*size_per_matrix].reshape(embedding_dim, dk)
            Wv = data[2*size_per_matrix:3*size_per_matrix].reshape(embedding_dim, dk)

            print("Wq (первые 5x5 элементов):\n", Wq[:5, :5])
            print("\nWk (первые 5x5 элементов):\n", Wk[:5, :5])
            print("\nWv (первые 5x5 элементов):\n", Wv[:5, :5])
            output_file = "/mnt/storage/product/ml-lab/baseModel/output_layer.pt"

            if not os.path.exists(output_file):
                print(f"[ERROR] Файл {output_file} не найден!")
            else:
                try:
                    with open(output_file, "rb") as f:
                        # Читаем заголовок
                        vocab_size = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
                        dk = int(np.frombuffer(f.read(4), dtype=np.int32)[0])
                        print(f"[INFO] Заголовок output_layer.pt: vocab_size={vocab_size}, dk={dk}")

                        # Считаем количество элементов
                        W_size = vocab_size * dk
                        b_size = vocab_size

                        # Читаем W_out
                        W_out = np.frombuffer(f.read(W_size * 4), dtype=np.float32).reshape(vocab_size, dk)
                        # Читаем b_out
                        b_out = np.frombuffer(f.read(b_size * 4), dtype=np.float32)

                        print(f"[INFO] W_out (первые 5x5 элементов):\n{W_out[:5, :5]}")
                        print(f"[INFO] b_out (первые 10 элементов): {b_out[:10]}")

                except Exception as e:
                    print(f"[ERROR] Ошибка при чтении output_layer.pt: {e}")

    except Exception as e:
        print(f"[ERROR] Ошибка при чтении weights.pt: {e}")

