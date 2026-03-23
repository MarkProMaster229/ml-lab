import re
import matplotlib.pyplot as plt
import numpy as np

def parse_logs(log_text: str):
    """
    Парсит строки вида:
    #Epoch  1 | Train Loss: 0.9854 | Val Loss: 0.4562 | Val Acc: 86.99%
    Возвращает кортеж (epochs, train_losses, val_losses, val_accs).
    """
    epochs = []
    train_losses = []
    val_losses = []
    val_accs = []

    pattern = r'Epoch\s+(\d+)\s*\|\s*Train Loss:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+)\s*\|\s*Val Acc:\s*([\d.]+)%'

    for line in log_text.splitlines():
        match = re.search(pattern, line)
        if match:
            epochs.append(int(match.group(1)))
            train_losses.append(float(match.group(2)))
            val_losses.append(float(match.group(3)))
            val_accs.append(float(match.group(4)))

    return epochs, train_losses, val_losses, val_accs

log_resnet = """
#Epoch  1 | Train Loss: 0.9854 | Val Loss: 0.4562 | Val Acc: 86.99%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.4000 | Val Loss: 0.4152 | Val Acc: 87.68%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3099 | Val Loss: 0.3289 | Val Acc: 90.05%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.2559 | Val Loss: 0.3136 | Val Acc: 90.43%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.2243 | Val Loss: 0.2719 | Val Acc: 91.95%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.1931 | Val Loss: 0.2549 | Val Acc: 92.28%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.1730 | Val Loss: 0.2513 | Val Acc: 92.44%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.1521 | Val Loss: 0.2437 | Val Acc: 92.49%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.1403 | Val Loss: 0.2439 | Val Acc: 92.48%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.1275 | Val Loss: 0.2828 | Val Acc: 91.99%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1165 | Val Loss: 0.2646 | Val Acc: 92.24%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1096 | Val Loss: 0.2579 | Val Acc: 92.55%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.0995 | Val Loss: 0.2828 | Val Acc: 92.49%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.0929 | Val Loss: 0.2709 | Val Acc: 92.55%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.0876 | Val Loss: 0.2916 | Val Acc: 92.56%
#Модель сохранена: cnn_model_epoch_15.pth
""" 

log_mobilenet = """
#Epoch  1 | Train Loss: 1.1325 | Val Loss: 0.5477 | Val Acc: 84.25%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.4790 | Val Loss: 0.4733 | Val Acc: 86.94%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3903 | Val Loss: 0.3612 | Val Acc: 89.77%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.3346 | Val Loss: 0.3245 | Val Acc: 90.35%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.3021 | Val Loss: 0.3038 | Val Acc: 90.68%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.2716 | Val Loss: 0.2841 | Val Acc: 91.16%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.2520 | Val Loss: 0.3179 | Val Acc: 90.69%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.2352 | Val Loss: 0.2777 | Val Acc: 91.56%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.2168 | Val Loss: 0.2801 | Val Acc: 91.56%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.2052 | Val Loss: 0.2672 | Val Acc: 91.91%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1914 | Val Loss: 0.2698 | Val Acc: 91.83%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1824 | Val Loss: 0.2501 | Val Acc: 92.17%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.1713 | Val Loss: 0.2654 | Val Acc: 91.88%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.1647 | Val Loss: 0.2573 | Val Acc: 92.28%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.1558 | Val Loss: 0.2382 | Val Acc: 92.74%
#Модель сохранена: cnn_model_epoch_15.pth
#Epoch 16 | Train Loss: 0.1523 | Val Loss: 0.2518 | Val Acc: 92.66%
#Модель сохранена: cnn_model_epoch_16.pth
#Epoch 17 | Train Loss: 0.1434 | Val Loss: 0.2441 | Val Acc: 92.74%
#Модель сохранена: cnn_model_epoch_17.pth
#Epoch 18 | Train Loss: 0.1383 | Val Loss: 0.2630 | Val Acc: 92.20%
#Модель сохранена: cnn_model_epoch_18.pth
#Epoch 19 | Train Loss: 0.1343 | Val Loss: 0.2363 | Val Acc: 93.03%
#Модель сохранена: cnn_model_epoch_19.pth
#Epoch 20 | Train Loss: 0.1262 | Val Loss: 0.2461 | Val Acc: 93.03%
#Модель сохранена: cnn_model_epoch_20.pth
#Epoch 21 | Train Loss: 0.1236 | Val Loss: 0.2660 | Val Acc: 92.12%
#Модель сохранена: cnn_model_epoch_21.pth
#Epoch 22 | Train Loss: 0.1218 | Val Loss: 0.2627 | Val Acc: 92.55%
#Модель сохранена: cnn_model_epoch_22.pth
#Epoch 23 | Train Loss: 0.1167 | Val Loss: 0.2479 | Val Acc: 92.63%
""" 

log_seresnet = """
#Epoch  1 | Train Loss: 1.4508 | Val Loss: 0.5244 | Val Acc: 85.04%
#Модель сохранена: cnn_model_epoch_1.pth
#Epoch  2 | Train Loss: 0.4297 | Val Loss: 0.3719 | Val Acc: 88.88%
#Модель сохранена: cnn_model_epoch_2.pth
#Epoch  3 | Train Loss: 0.3424 | Val Loss: 0.3682 | Val Acc: 89.00%
#Модель сохранена: cnn_model_epoch_3.pth
#Epoch  4 | Train Loss: 0.2957 | Val Loss: 0.3323 | Val Acc: 89.97%
#Модель сохранена: cnn_model_epoch_4.pth
#Epoch  5 | Train Loss: 0.2599 | Val Loss: 0.2791 | Val Acc: 91.13%
#Модель сохранена: cnn_model_epoch_5.pth
#Epoch  6 | Train Loss: 0.2317 | Val Loss: 0.2870 | Val Acc: 91.49%
#Модель сохранена: cnn_model_epoch_6.pth
#Epoch  7 | Train Loss: 0.2105 | Val Loss: 0.3122 | Val Acc: 90.82%
#Модель сохранена: cnn_model_epoch_7.pth
#Epoch  8 | Train Loss: 0.1951 | Val Loss: 0.2713 | Val Acc: 92.02%
#Модель сохранена: cnn_model_epoch_8.pth
#Epoch  9 | Train Loss: 0.1782 | Val Loss: 0.2380 | Val Acc: 92.72%
#Модель сохранена: cnn_model_epoch_9.pth
#Epoch 10 | Train Loss: 0.1636 | Val Loss: 0.2543 | Val Acc: 92.24%
#Модель сохранена: cnn_model_epoch_10.pth
#Epoch 11 | Train Loss: 0.1533 | Val Loss: 0.2500 | Val Acc: 92.41%
#Модель сохранена: cnn_model_epoch_11.pth
#Epoch 12 | Train Loss: 0.1434 | Val Loss: 0.2589 | Val Acc: 92.41%
#Модель сохранена: cnn_model_epoch_12.pth
#Epoch 13 | Train Loss: 0.1370 | Val Loss: 0.2367 | Val Acc: 92.91%
#Модель сохранена: cnn_model_epoch_13.pth
#Epoch 14 | Train Loss: 0.1272 | Val Loss: 0.2691 | Val Acc: 92.51%
#Модель сохранена: cnn_model_epoch_14.pth
#Epoch 15 | Train Loss: 0.1192 | Val Loss: 0.2576 | Val Acc: 92.52%
#Модель сохранена: cnn_model_epoch_15.pth
#Epoch 16 | Train Loss: 0.1127 | Val Loss: 0.2795 | Val Acc: 92.05%
#Модель сохранена: cnn_model_epoch_16.pth
#Epoch 17 | Train Loss: 0.1054 | Val Loss: 0.2685 | Val Acc: 92.43%
#Модель сохранена: cnn_model_epoch_17.pth
#Epoch 18 | Train Loss: 0.1002 | Val Loss: 0.2932 | Val Acc: 91.41%
#Модель сохранена: cnn_model_epoch_18.pth
#Epoch 19 | Train Loss: 0.0922 | Val Loss: 0.2923 | Val Acc: 92.19%
#Модель сохранена: cnn_model_epoch_19.pth
#Epoch 20 | Train Loss: 0.0908 | Val Loss: 0.2934 | Val Acc: 92.11%
#Модель сохранена: cnn_model_epoch_20.pth
"""  

resnet = parse_logs(log_resnet)
mobilenet = parse_logs(log_mobilenet)
seresnet = parse_logs(log_seresnet)


models = {
    'ResNet18': resnet,
    'MobileNetV2': mobilenet,
    'SE_ResNet18': seresnet
}
colors = {'ResNet18': 'blue', 'MobileNetV2': 'green', 'SE_ResNet18': 'red'}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Сравнение архитектур: динамика обучения', fontsize=16, fontweight='bold')

# 1. Ошибка обучения
ax1 = axes[0, 0]
for name, (epochs, train, val, acc) in models.items():
    ax1.plot(epochs, train, label=name, color=colors[name], linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Ошибка обучения', fontsize=12)
ax1.set_title('Динамика ошибки на обучающей выборке', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Ошибка валидации
ax2 = axes[0, 1]
for name, (epochs, train, val, acc) in models.items():
    ax2.plot(epochs, val, label=name, color=colors[name], linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Ошибка валидации', fontsize=12)
ax2.set_title('Динамика ошибки на валидационной выборке', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Точность валидации
ax3 = axes[1, 0]
for name, (epochs, train, val, acc) in models.items():
    ax3.plot(epochs, acc, label=name, color=colors[name], linewidth=2, marker='^', markersize=4)
ax3.set_xlabel('Эпоха', fontsize=12)
ax3.set_ylabel('Точность валидации (%)', fontsize=12)
ax3.set_title('Динамика точности на валидационной выборке', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Разрыв (Val Loss - Train Loss) – анализ переобучения
ax4 = axes[1, 1]
for name, (epochs, train, val, acc) in models.items():
    gap = [v - t for v, t in zip(val, train)]
    ax4.plot(epochs, gap, label=name, color=colors[name], linewidth=2, marker='o', markersize=4)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Нулевой разрыв')
ax4.set_xlabel('Эпоха', fontsize=12)
ax4.set_ylabel('Ошибка валидации − Ошибка обучения', fontsize=12)
ax4.set_title('Анализ переобучения', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 80)
print("АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
print("=" * 80)

for name, (epochs, train, val, acc) in models.items():
    best_acc_idx = np.argmax(acc)
    best_val_idx = np.argmin(val)
    print(f"\n{name}:")
    print(f"  Лучшая точность валидации: {acc[best_acc_idx]:.2f}% на эпохе {epochs[best_acc_idx]}")
    print(f"  Минимальная ошибка валидации: {val[best_val_idx]:.4f} на эпохе {epochs[best_val_idx]}")
    print(f"  Количество эпох: {len(epochs)}")