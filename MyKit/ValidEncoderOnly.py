import re
import matplotlib.pyplot as plt
import numpy as np

def parse_my_logs(log_text: str):
    """Универсальный парсер для двух форматов:
    1) Epoch X Train Loss: ... / Epoch X Loss in Validation data ...
    2) Epoch X loss in train ... / Epoch X loss in Valid ...
    """
    epochs = []
    train_losses = []
    val_losses = []
    
    lines = log_text.splitlines()
    
    for line in lines:
        line = line.lstrip('#').strip()
        

        train_match = re.search(r'Epoch (\d+) Train Loss: ([\d.]+)', line)
        if not train_match:
            train_match = re.search(r'Epoch (\d+) loss in train ([\d.]+)', line)
        
        if train_match:
            epoch = int(train_match.group(1))
            train_loss = float(train_match.group(2))
            epochs.append(epoch)
            train_losses.append(train_loss)
        
        val_match = re.search(r'Epoch (\d+) Loss in Validation data ([\d.]+)', line)
        if not val_match:
            val_match = re.search(r'Epoch (\d+) loss in Valid ([\d.]+)', line)
        
        if val_match:
            val_loss = float(val_match.group(2))
            val_losses.append(val_loss)
    
    return epochs, train_losses, val_losses
# BERT без LoRA (full fine-tuning)
bert_no_lora_log = """
#Epoch 0 Train Loss: 0.8686387965253732
#Epoch 0 Loss in Validation data 0.769134180771338
#Epoch 1 Train Loss: 0.7611743432405007
#Epoch 1 Loss in Validation data 0.746833496802562
#Epoch 2 Train Loss: 0.675213236938005
#Epoch 2 Loss in Validation data 0.7892871507116266
#Epoch 3 Train Loss: 0.595656931611293
#Epoch 3 Loss in Validation data 0.7977193585420782
#Epoch 4 Train Loss: 0.49758275126954415
#Epoch 4 Loss in Validation data 0.887513331464819
#Epoch 5 Train Loss: 0.3946617727069214
#Epoch 5 Loss in Validation data 0.9465082672399443
#Epoch 6 Train Loss: 0.3112003424565901
#Epoch 6 Loss in Validation data 1.100050000247319
#Epoch 7 Train Loss: 0.2355832863290937
#Epoch 7 Loss in Validation data 1.274325226232208
#Epoch 8 Train Loss: 0.18996810883570897
#Epoch 8 Loss in Validation data 1.1774581715659667
#Epoch 9 Train Loss: 0.15845887185246388
#Epoch 9 Loss in Validation data 1.3258019001097292
#Epoch 10 Train Loss: 0.12964913105513717
#Epoch 10 Loss in Validation data 1.6574379938679773
#Epoch 11 Train Loss: 0.1184352965175647
#Epoch 11 Loss in Validation data 1.5163718392267018
#Epoch 12 Train Loss: 0.10106043583907132
#Epoch 12 Loss in Validation data 1.6204950932532902
#Epoch 13 Train Loss: 0.09975709741751577
#Epoch 13 Loss in Validation data 1.499698380192402
#Epoch 14 Train Loss: 0.093336668817841
#Epoch 14 Loss in Validation data 1.3822113348617897
#Epoch 15 Train Loss: 0.08832583144544044
#Epoch 15 Loss in Validation data 1.6743186236031957
#Epoch 16 Train Loss: 0.07559707110726638
#Epoch 16 Loss in Validation data 1.8559099168161157
#Epoch 17 Train Loss: 0.07997204229968315
#Epoch 17 Loss in Validation data 1.849636017370063
#Epoch 18 Train Loss: 0.07319258857468558
#Epoch 18 Loss in Validation data 1.7581849822623503
#Epoch 19 Train Loss: 0.06875287767275788
#Epoch 19 Loss in Validation data 1.6340184822879933
"""

# BERT с LoRA
bert_lora_log = """

#Epoch 0 loss in train 0.9268037862667358
#Epoch 0 loss in Valid 0.9856747420210588
#Epoch 1 loss in train 0.8913424615447617
#Epoch 1 loss in Valid 0.9158578201344139
#Epoch 2 loss in train 0.8698545825031294
#Epoch 2 loss in Valid 0.8792771916640433
#Epoch 3 loss in train 0.8371046626901801
#Epoch 3 loss in Valid 0.7641685777588895
#Epoch 4 loss in train 0.8092562085818431
#Epoch 4 loss in Valid 0.8229999385382
#Epoch 5 loss in train 0.7869190592205336
#Epoch 5 loss in Valid 0.7866415820623699
#Epoch 6 loss in train 0.7665394056366074
#Epoch 6 loss in Valid 0.8540124422625491
#Epoch 7 loss in train 0.749155752732896
#Epoch 7 loss in Valid 0.7070749998092651
#Epoch 8 loss in train 0.7298482119601478
#Epoch 8 loss in Valid 0.720144293810192
#Epoch 9 loss in train 0.7147667670075699
#Epoch 9 loss in Valid 0.6904838006747397
#Epoch 10 loss in train 0.7015375231237272
#Epoch 10 loss in Valid 0.7583794185989782
#Epoch 11 loss in train 0.6888231706604743
#Epoch 11 loss in Valid 0.8473481159461173
#Epoch 12 loss in train 0.6744434226045655
#Epoch 12 loss in Valid 0.6963813351957422
#Epoch 13 loss in train 0.6713746627667935
#Epoch 13 loss in Valid 0.6966002458020261
#Epoch 14 loss in train 0.6493793887684028
#Epoch 14 loss in Valid 0.6894897898953212
#Epoch 15 loss in train 0.6380676526063542
#Epoch 15 loss in Valid 0.7037043759697362
#Epoch 16 loss in train 0.6354927422431151
#Epoch 16 loss in Valid 0.6870511283999995
#Epoch 17 loss in train 0.6155846477752771
#Epoch 17 loss in Valid 0.7176429503842404
#Epoch 18 loss in train 0.6124847058329309
#Epoch 18 loss in Valid 0.7578575469945606
#Epoch 19 loss in train 0.5969803444269804
#Epoch 19 loss in Valid 0.824189891940669
"""

# DistilBERT с LoRA
distilbert_lora_log = """
#Epoch 0 loss in train 0.9144839738086138
#Epoch 0 loss in Valid 0.9012686955301386
#Epoch 1 loss in train 0.8720600287022747
#Epoch 1 loss in Valid 0.8137206654799612
#Epoch 2 loss in train 0.8450387906505478
#Epoch 2 loss in Valid 0.7938372963353207
#Epoch 3 loss in train 0.8304082767609121
#Epoch 3 loss in Valid 0.7822490585477728
#Epoch 4 loss in train 0.8218647722930769
#Epoch 4 loss in Valid 0.7740178743475362
#Epoch 5 loss in train 0.8122451209194332
#Epoch 5 loss in Valid 0.8751911238620156
#Epoch 6 loss in train 0.8030335002927048
#Epoch 6 loss in Valid 0.7790471315383911
#Epoch 7 loss in train 0.7998993278420944
#Epoch 7 loss in Valid 0.7613500482157657
#Epoch 8 loss in train 0.7917967596602934
#Epoch 8 loss in Valid 0.7891154101020411
#Epoch 9 loss in train 0.7772026442799005
#Epoch 9 loss in Valid 0.8104652825154757
#Epoch 10 loss in train 0.7748061650853499
#Epoch 10 loss in Valid 0.7402110601726332
#Epoch 11 loss in train 0.7684638743423806
#Epoch 11 loss in Valid 0.7508765317891773
#Epoch 12 loss in train 0.7598204258443667
#Epoch 12 loss in Valid 0.7105655403513658
#Epoch 13 loss in train 0.7526728668761747
#Epoch 13 loss in Valid 0.7288588865807182
#Epoch 14 loss in train 0.7470956186284101
#Epoch 14 loss in Valid 0.7330299913883209
#Epoch 15 loss in train 0.7375644864321627
#Epoch 15 loss in Valid 0.751100701721091
#Epoch 16 loss in train 0.732530345870284
#Epoch 16 loss in Valid 0.769866921399769
#Epoch 17 loss in train 0.7242152105005011
#Epoch 17 loss in Valid 0.7093124938638586
#Epoch 18 loss in train 0.7182778385757094
#Epoch 18 loss in Valid 0.7614692888761821
#Epoch 19 loss in train 0.7149477876568537
#Epoch 19 loss in Valid 0.7259659908319774
"""

# XLM-RoBERTa с LoRA
xlm_roberta_log = """
#Epoch 0 loss in train 0.9158469306783757
#Epoch 0 loss in Valid 0.7410750765549509
#Epoch 1 loss in train 0.737323088528323
#Epoch 1 loss in Valid 0.6086877993258991
#Epoch 2 loss in train 0.6775894329115186
#Epoch 2 loss in Valid 0.5845026852268922
#Epoch 3 loss in train 0.643513333891835
#Epoch 3 loss in Valid 0.5461553405774268
#Epoch 4 loss in train 0.6184801802529197
#Epoch 4 loss in Valid 0.5396700407329359
#Epoch 5 loss in train 0.5931856373050471
#Epoch 5 loss in Valid 0.5236105121868221
#Epoch 6 loss in train 0.5723478227272975
#Epoch 6 loss in Valid 0.5343811045351782
#Epoch 7 loss in train 0.553374152562505
#Epoch 7 loss in Valid 0.574058389977405
#Epoch 8 loss in train 0.5281864021622307
#Epoch 8 loss in Valid 0.5165252160084876
#Epoch 9 loss in train 0.5149054350592333
#Epoch 9 loss in Valid 0.6286481352228868
#Epoch 10 loss in train 0.49756711899290423
#Epoch 10 loss in Valid 0.5128102786839008
#Epoch 11 loss in train 0.48466512180601634
#Epoch 11 loss in Valid 0.5384572548301596
#Epoch 12 loss in train 0.47197424308663455
#Epoch 12 loss in Valid 0.587054130645763
#Epoch 13 loss in train 0.44659905255875404
#Epoch 13 loss in Valid 0.5688826889289837
#Epoch 14 loss in train 0.4435478064495884
#Epoch 14 loss in Valid 0.8040276240361365
#Epoch 15 loss in train 0.4270620637853493
#Epoch 15 loss in Valid 0.6674159624074635
#Epoch 16 loss in train 0.41355355001132976
#Epoch 16 loss in Valid 0.6027880973721805
#Epoch 17 loss in train 0.4019455197749344
#Epoch 17 loss in Valid 0.5945103256718108
#Epoch 18 loss in train 0.3846266034595786
#Epoch 18 loss in Valid 0.954741687366837
#Epoch 19 loss in train 0.373605940382061
#Epoch 19 loss in Valid 0.6239002806771743
"""

#нет
classification_small_log = """

"""

models = {
    "BERT (full fine-tune)": parse_my_logs(bert_no_lora_log),
    "BERT + LoRA": parse_my_logs(bert_lora_log),
    "DistilBERT + LoRA": parse_my_logs(distilbert_lora_log),
    "XLM-RoBERTa + LoRA": parse_my_logs(xlm_roberta_log),
    "ClassificationSmall": parse_my_logs(classification_small_log),
}

models = {name: (e, t, v) for name, (e, t, v) in models.items() if e and t}

for name, (e, t, v) in models.items():
    print(f"{name}: epochs={len(e)}, train_losses={len(t)}, val_losses={len(v)}")


colors = {
    "BERT (full fine-tune)": "#1f77b4",
    "BERT + LoRA": "#2ca02c",
    "DistilBERT + LoRA": "#ff7f0e",
    "XLM-RoBERTa + LoRA": "#d62728",
    "ClassificationSmall": "#9467bd",
}

# График 1: все модели (train и val)
fig, ax = plt.subplots(figsize=(12, 7))
for name, (epochs, train_losses, val_losses) in models.items():
    color = colors.get(name, "#333333")
    ax.plot(epochs, train_losses, 
            label=f"{name} (train)", 
            color=color, 
            linestyle="-", 
            marker='o', 
            markersize=3,
            linewidth=1.5)
    if val_losses and len(val_losses) == len(epochs):
        ax.plot(epochs, val_losses, 
                label=f"{name} (val)", 
                color=color, 
                linestyle="--", 
                marker='s', 
                markersize=3,
                linewidth=1.5)

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Training and Validation Loss (all models)", fontsize=14)
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_all_models.png", dpi=150)
plt.show()

# График 2: без full fine-tune
fig2, ax2 = plt.subplots(figsize=(10, 6))
best_names = ["XLM-RoBERTa + LoRA", "BERT + LoRA", "DistilBERT + LoRA"]
for name in best_names:
    if name in models:
        epochs, train_losses, val_losses = models[name]
        color = colors.get(name, "#333333")
        ax2.plot(epochs, train_losses, 
                 label=f"{name} (train)", 
                 color=color, 
                 linestyle="-", 
                 marker='o', 
                 markersize=4)
        if val_losses and len(val_losses) == len(epochs):
            ax2.plot(epochs, val_losses, 
                     label=f"{name} (val)", 
                     color=color, 
                     linestyle="--", 
                     marker='s', 
                     markersize=4)

ax2.set_xlabel("Epoch", fontsize=12)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Loss Comparison (LoRA models)", fontsize=14)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("loss_lora_models.png", dpi=150)
plt.show()

# График 3: только validation loss для всех
fig3, ax3 = plt.subplots(figsize=(10, 6))
for name, (epochs, train_losses, val_losses) in models.items():
    if val_losses and len(val_losses) == len(epochs):
        color = colors.get(name, "#333333")
        ax3.plot(epochs, val_losses, 
                 label=f"{name}", 
                 color=color, 
                 marker='s', 
                 markersize=5,
                 linewidth=2)
ax3.set_xlabel("Epoch", fontsize=12)
ax3.set_ylabel("Validation Loss", fontsize=12)
ax3.set_title("Validation Loss Comparison", fontsize=14)
ax3.legend(loc="upper right", fontsize=9)
ax3.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("validation_loss_only.png", dpi=150)
plt.show()
