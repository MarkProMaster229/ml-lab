import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Устанавливаем seed для воспроизводимости
np.random.seed(42)
n_samples = 1000

# 1. Генерируем случайные координаты X и Y от -10 до 10
X_coords = np.random.uniform(-10, 10, n_samples)
Y_coords = np.random.uniform(-10, 10, n_samples)

# 2. Генерируем признак Z — это просто случайный шум (модель должна его проигнорировать)
Z_noise = np.random.uniform(-5, 5, n_samples)

# 3. Считаем квадрат расстояния до центра по теореме Пифагора: R^2 = X^2 + Y^2
distance_sq = X_coords**2 + Y_coords**2

# 4. Хитрая математическая логика для классов:
# Если точка внутри радиуса 5 — это класс 0. Если между 5 и 8.5 — класс 1. Если дальше — класс 0.
target = np.zeros(n_samples, dtype=int)
# Кольцо (Класс 1)
target[(distance_sq >= 5**2) & (distance_sq <= 8.5**2)] = 1

# Добавим капельку шума в таргет (типа "ошибка измерения" в 2% случаев)
noise_mask = np.random.rand(n_samples) < 0.02
target[noise_mask] = 1 - target[noise_mask]

# 5. Собираем всё в DataFrame и сохраняем в CSV
df = pd.DataFrame({
    'Feature_X': X_coords,
    'Feature_Y': Y_coords,
    'Feature_Z_Noise': Z_noise,
    'Class': target
})

df.to_csv('secret_math_pattern.csv', index=False)
print("Файл 'secret_math_pattern.csv' успешно создан!")

# 6. Визуализируем, чтобы ты глазами увидел неочевидную для алгоритмов структуру
plt.figure(figsize=(7, 7))
plt.scatter(df[df['Class'] == 0]['Feature_X'], df[df['Class'] == 0]['Feature_Y'], c='blue', alpha=0.6, label='Класс 0 (Центр и Окраина)')
plt.scatter(df[df['Class'] == 1]['Feature_X'], df[df['Class'] == 1]['Feature_Y'], c='red', alpha=0.6, label='Класс 1 (Секретное кольцо)')
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.title("Математическая закономерность: Кольцо")
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig('pattern_visualization.png')
print("График сохранен в 'pattern_visualization.png'. Посмотри на него перед обучением!")
