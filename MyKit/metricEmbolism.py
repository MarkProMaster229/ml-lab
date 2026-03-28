import re
import matplotlib.pyplot as plt
import numpy as np
def parse_logs(log_text: str):
    epochs = []
    train_losses = []
    val_losses = []

    lines = log_text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match_epoch = re.search(r'#\s*Эпоха\s*\[(\d+)/\d+\],\s*Средняя ошибка:\s*([\d.]+)', line)
        if match_epoch:
            epoch = int(match_epoch.group(1))
            train_loss = float(match_epoch.group(2))
            val_loss = None
            j = i + 1
            while j < len(lines) and val_loss is None:
                next_line = lines[j].strip()
                match_val = re.search(r'#\s*Валидация:\s*Средняя ошибка:\s*([\d.]+)', next_line)
                if match_val:
                    val_loss = float(match_val.group(1))
                j += 1
            if val_loss is not None:
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            i = j
        else:
            i += 1

    return epochs, train_losses, val_losses
log_unet = """
#Эпоха [1/25], Средняя ошибка: 0.9650
#Валидация: Средняя ошибка: 0.9403
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/25], Средняя ошибка: 0.8257
#Валидация: Средняя ошибка: 0.8836
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/25], Средняя ошибка: 0.7918
#Валидация: Средняя ошибка: 0.8887
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/25], Средняя ошибка: 0.7571
#Валидация: Средняя ошибка: 0.8625
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/25], Средняя ошибка: 0.7316
#Валидация: Средняя ошибка: 0.8713
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/25], Средняя ошибка: 0.7029
#Валидация: Средняя ошибка: 0.8787
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/25], Средняя ошибка: 0.7010
#Валидация: Средняя ошибка: 0.8599
#Модель сохранена: cnn_model_epoch_7.pth
#Эпоха [8/25], Средняя ошибка: 0.6846
#Валидация: Средняя ошибка: 0.8981
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/25], Средняя ошибка: 0.6563
#Валидация: Средняя ошибка: 0.8524
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/25], Средняя ошибка: 0.6472
#Валидация: Средняя ошибка: 0.8558
"""

log_attention_unet = """
# Эпоха [1/10], Средняя ошибка: 0.9622
# Валидация: Средняя ошибка: 0.9011
# Модель сохранена: cnn_model_epoch_1.pth
# Эпоха [2/10], Средняя ошибка: 0.8077
# Валидация: Средняя ошибка: 0.8787
# Модель сохранена: cnn_model_epoch_2.pth
# Эпоха [3/10], Средняя ошибка: 0.7757
# Валидация: Средняя ошибка: 0.8779
# Модель сохранена: cnn_model_epoch_3.pth
# Эпоха [4/10], Средняя ошибка: 0.7468
# Валидация: Средняя ошибка: 0.8630
# Модель сохранена: cnn_model_epoch_4.pth
# Эпоха [5/10], Средняя ошибка: 0.7446
# Валидация: Средняя ошибка: 0.8669
# Модель сохранена: cnn_model_epoch_5.pth
# Эпоха [6/10], Средняя ошибка: 0.7160
# Валидация: Средняя ошибка: 0.8544
# Модель сохранена: cnn_model_epoch_6.pth
# Эпоха [7/10], Средняя ошибка: 0.6911
# Валидация: Средняя ошибка: 0.8574
# Модель сохранена: cnn_model_epoch_7.pth
# Эпоха [8/10], Средняя ошибка: 0.6712
# Валидация: Средняя ошибка: 0.8516
# Модель сохранена: cnn_model_epoch_8.pth
# Эпоха [9/10], Средняя ошибка: 0.6582
# Валидация: Средняя ошибка: 0.8464
# Модель сохранена: cnn_model_epoch_9.pth
# Эпоха [10/10], Средняя ошибка: 0.6524
# Валидация: Средняя ошибка: 0.8642
# Модель сохранена: cnn_model_epoch_10.pth
"""
log_3d_unet_vgg = """
#Эпоха [1/10], Средняя ошибка: 0.9706
#Валидация: Средняя ошибка: 0.9211
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/10], Средняя ошибка: 0.8998
#Валидация: Средняя ошибка: 0.8821
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/10], Средняя ошибка: 0.8862
#Валидация: Средняя ошибка: 0.8667
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/10], Средняя ошибка: 0.8757
#Валидация: Средняя ошибка: 0.8721
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/10], Средняя ошибка: 0.8651
#Валидация: Средняя ошибка: 0.8341
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/10], Средняя ошибка: 0.8562
#Валидация: Средняя ошибка: 0.8674
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/10], Средняя ошибка: 0.8474
#Валидация: Средняя ошибка: 0.8620
#Модель сохранена: cnn_model_epoch_7.pth
#Эпоха [8/10], Средняя ошибка: 0.8402
#Валидация: Средняя ошибка: 0.8456
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/10], Средняя ошибка: 0.8336
#Валидация: Средняя ошибка: 0.8183
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/10], Средняя ошибка: 0.8262
#Валидация: Средняя ошибка: 0.8879
#Модель сохранена: cnn_model_epoch_10.pth
"""

log_3d_attention_unet_vgg = """
#Эпоха [1/10], Средняя ошибка: 0.9451
#Валидация: Средняя ошибка: 0.8815
#Модель сохранена: cnn_model_epoch_1.pth
#Эпоха [2/10], Средняя ошибка: 0.8940
#Валидация: Средняя ошибка: 0.8626
#Модель сохранена: cnn_model_epoch_2.pth
#Эпоха [3/10], Средняя ошибка: 0.8750
#Валидация: Средняя ошибка: 0.8640
#Модель сохранена: cnn_model_epoch_3.pth
#Эпоха [4/10], Средняя ошибка: 0.8619
#Валидация: Средняя ошибка: 0.8537
#Модель сохранена: cnn_model_epoch_4.pth
#Эпоха [5/10], Средняя ошибка: 0.8470
#Валидация: Средняя ошибка: 0.8409
#Модель сохранена: cnn_model_epoch_5.pth
#Эпоха [6/10], Средняя ошибка: 0.8394
#Валидация: Средняя ошибка: 0.8483
#Модель сохранена: cnn_model_epoch_6.pth
#Эпоха [7/10], Средняя ошибка: 0.8308
#Валидация: Средняя ошибка: 0.8381
##Модель сохранена: cnn_model_epoch_7.pth
#Эпоха [8/10], Средняя ошибка: 0.8312
#Валидация: Средняя ошибка: 0.8577
#Модель сохранена: cnn_model_epoch_8.pth
#Эпоха [9/10], Средняя ошибка: 0.8184
#Валидация: Средняя ошибка: 0.8448
#Модель сохранена: cnn_model_epoch_9.pth
#Эпоха [10/10], Средняя ошибка: 0.8126
#Валидация: Средняя ошибка: 0.8373
#Модель сохранена: cnn_model_epoch_10.pth
"""
# ---------------------------------------------------------------------------

unet = parse_logs(log_unet)
attention_unet = parse_logs(log_attention_unet)
unet_3d_vgg = parse_logs(log_3d_unet_vgg)
attention_unet_3d_vgg = parse_logs(log_3d_attention_unet_vgg)

models = {
    'модель типа U-NET': unet,
    'U net с механизмом внимания': attention_unet,
    '3D U-Net с VGG-подобным энкодером': unet_3d_vgg,
    '3D Attention U-Net с VGG-подобным энкодером': attention_unet_3d_vgg
}

# Цвета
colors = {
    'модель типа U-NET': 'blue',
    'U net с механизмом внимания': 'green',
    '3D U-Net с VGG-подобным энкодером': 'red',
    '3D Attention U-Net с VGG-подобным энкодером': 'orange'
}

# Графики (убираем график точности, оставляем два графика для loss и разрыв)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1x3 вместо 2x2
fig.suptitle('Динамика обучения', fontsize=16, fontweight='bold')

# 1. Ошибка обучения
ax1 = axes[0]
for name, (epochs, train, val) in models.items():
    ax1.plot(epochs, train, label=name, color=colors[name], linewidth=2, marker='o', markersize=4)
ax1.set_xlabel('Эпоха', fontsize=12)
ax1.set_ylabel('Ошибка обучения', fontsize=12)
ax1.set_title('Динамика ошибки на обучающей выборке', fontsize=12, fontweight='bold')

ax1.grid(True, alpha=0.3)

# 2. Ошибка валидации
ax2 = axes[1]
for name, (epochs, train, val) in models.items():
    ax2.plot(epochs, val, label=name, color=colors[name], linewidth=2, marker='s', markersize=4)
ax2.set_xlabel('Эпоха', fontsize=12)
ax2.set_ylabel('Ошибка валидации', fontsize=12)
ax2.set_title('Динамика ошибки на валидационной выборке', fontsize=12, fontweight='bold')

ax2.grid(True, alpha=0.3)

# 3. Разрыв (Val Loss - Train Loss)
ax3 = axes[2]
for name, (epochs, train, val) in models.items():
    gap = [v - t for v, t in zip(val, train)]
    ax3.plot(epochs, gap, label=name, color=colors[name], linewidth=2, marker='^', markersize=4)
ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7, label='Нулевой разрыв')
ax3.set_xlabel('Эпоха', fontsize=12)
ax3.set_ylabel('Ошибка валидации − Ошибка обучения', fontsize=12)
ax3.set_title('Анализ переобучения', fontsize=12, fontweight='bold')
ax3.legend(bbox_to_anchor=(0.5, -0.2), loc='upper center', fontsize=9, ncol=2)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 80)
print("АНАЛИЗ РЕЗУЛЬТАТОВ ОБУЧЕНИЯ")
print("=" * 80)

for name, (epochs, train, val) in models.items():
    best_val_idx = np.argmin(val)
    print(f"\n{name}:")
    print(f"  Минимальная ошибка валидации: {val[best_val_idx]:.4f} на эпохе {epochs[best_val_idx]}")
    print(f"  Количество эпох: {len(epochs)}")