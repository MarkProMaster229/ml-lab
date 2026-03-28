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

""" 

log_mobilenet = """

""" 

log_seresnet = """

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