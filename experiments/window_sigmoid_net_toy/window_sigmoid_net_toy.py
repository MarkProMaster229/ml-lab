import numpy as np

#предложения
sentences = [

]



vocab = {}
for sent in sentences:
    for word in sent.split():
        if word not in vocab:
            vocab[word] = len(vocab)

# токены для обучения
tokenized_sentencesTRAINER = [[vocab[word] for word in sent.split()] for sent in sentences]
print("Словарь:", vocab)
print("Токенизированные предложения:", tokenized_sentencesTRAINER)


import random

tokenized_shuffled_words = [
    random.sample(sentence, len(sentence))  # случайное перемешивание
    for sentence in tokenized_sentencesTRAINER
]
print("данные обучения предложения:", tokenized_shuffled_words)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

input_size = 5
hidden_size = 100
wekrtor_size = np.random.rand(hidden_size, input_size)
bias_hidden = np.random.randn(hidden_size, 1)

hidden_activations = []

for sentence in tokenized_sentencesTRAINER:
    # Пробегаем "окном" длины input_size
    for i in range(len(sentence) - input_size + 1):
        x_tokens = sentence[i:i+input_size]  # берем 5 токенов
        x = np.array(x_tokens, dtype=float).reshape(input_size, 1)  # shape (input_size,1)
        z = np.dot(wekrtor_size, x) + bias_hidden  # shape (hidden_size,1)
        a = sigmoid(z)
        hidden_activations.append(a)

print("активация скрытого слоя")
for a in hidden_activations:
    print(a)

#кол-во нейронов на выходном слое
outputSize = 10

# веса выходного слоя (каждый выходной нейрон соединен с нейронами скрытого слоя)
weights_output = np.random.randn(outputSize, hidden_size)
# смещение для выходного нейрона
bias_output = np.random.randn(outputSize, 1)

# активации выходного слоя
output_activations = []
for a_hidden in hidden_activations:
    z_output = np.dot(weights_output, a_hidden) + bias_output
    a_output = sigmoid(z_output)
    output_activations.append(a_output)

print("Активации выходного слоя (предсказания):")
for out in output_activations:
    print(out)

def mse_loss(y_pred, y_true):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Обратное распространение
def backprop(x, y):
    z_hidden = np.dot(wekrtor_size, x) + bias_hidden
    a_hidden = sigmoid(z_hidden)
    z_output = np.dot(weights_output, a_hidden) + bias_output
    a_output = sigmoid(z_output)

    delta_output = (a_output - y) * a_output * (1 - a_output)
    grad_w_output = np.dot(delta_output, a_hidden.T)
    grad_b_output = delta_output

    delta_hidden = np.dot(weights_output.T, delta_output) * a_hidden * (1 - a_hidden)
    grad_w_hidden = np.dot(delta_hidden, x.T)
    grad_b_hidden = delta_hidden

    return grad_w_hidden, grad_b_hidden, grad_w_output, grad_b_output

eta = 0.5  # скорость обучения

# Исправляем backprop под input_size >1
# для backprop оставляем один токен на выход
weights_output = np.random.randn(1, hidden_size)
bias_output = np.random.randn(1, 1)

# обучаем на последнем токене окна
for epoch in range(10000):
    for sentence_input, sentence_target in zip(tokenized_shuffled_words, tokenized_sentencesTRAINER):
        for i in range(len(sentence_input) - input_size + 1):
            x_tokens = sentence_input[i:i+input_size]
            y_token = sentence_target[i+input_size-1]  # берем последний токен окна как правильный
            x = np.array(x_tokens, dtype=float).reshape(input_size, 1)
            y = np.array([[y_token]], dtype=float)

            gw_h, gb_h, gw_o, gb_o = backprop(x, y)

            wekrtor_size -= eta * gw_h
            bias_hidden -= eta * gb_h
            weights_output -= eta * gw_o
            bias_output -= eta * gb_o

def softmax(z):
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / exp_z.sum(axis=0, keepdims=True)

output_size = len(vocab)
weights_output = np.random.rand(output_size, hidden_size)
bias_output = np.random.randn(output_size, 1)

for sentence in tokenized_sentencesTRAINER:
    for i in range(len(sentence) - input_size + 1):
        x_tokens = sentence[i:i+input_size]
        #нормализация входа
        x = np.array(x_tokens, dtype=float).reshape(input_size, 1)

        z_hidden = np.dot(wekrtor_size, x) + bias_hidden
        a_hidden = sigmoid(z_hidden)

        z_output = np.dot(weights_output, a_hidden) + bias_output
        a_output = softmax(z_output)

        print("Токен:", x_tokens)
        print("Логиты выходного слоя:", z_output.ravel())
        print("Активация (вероятности):", a_output.ravel())

inv_vocab = {v: k for k, v in vocab.items()}

start_sequence = ['Сынок', ',', 'ты', 'будешь', 'славный']
generated_tokens = [vocab[w] for w in start_sequence]
max_steps = 10

T = 0.5  # температура

for _ in range(max_steps):
    x = np.array(generated_tokens[-input_size:], dtype=float).reshape(input_size, 1)
    z_hidden = np.dot(wekrtor_size, x) + bias_hidden
    a_hidden = sigmoid(z_hidden)
    z_output = np.dot(weights_output, a_hidden) + bias_output

    # применяем температуру перед softmax
    z_scaled = z_output / T
    a_output = softmax(z_scaled)

    # теперь сэмплинг по распределению
    next_token = np.random.choice(len(a_output), p=a_output.ravel())
    generated_tokens.append(next_token)

generated_words = [inv_vocab[t] for t in generated_tokens]
print("Сгенерированная последовательность:", " ".join(generated_words))
