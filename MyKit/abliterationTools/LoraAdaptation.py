import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import partial
import os

# ==============================
# Конфигурация
# ==============================
BASE_MODEL_PATH = "/home/chelovek/Music/123/"
TRAIN_JSON_PATH = "/home/chelovek/Desktop/trainNEW321.json"
VALID_JSON_PATH = "/home/chelovek/Desktop/validNEW321.json"
OUTPUT_DIR = "/home/chelovek/Music/modelWork/ml-lab/mayBEthisfinaly"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_LENGTH = 512
PHYSICAL_BATCH_SIZE = 3
TARGET_BATCH_SIZE = 12
GRAD_ACCUM_STEPS = max(1, TARGET_BATCH_SIZE // PHYSICAL_BATCH_SIZE)

EPOCHS = 10
LEARNING_RATE = 5e-5

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.3
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj", "out_proj", "x_proj"]

# ==============================
# Датасет и коллатор
# ==============================
class UniversalDialogDataset(Dataset):
    def __init__(
        self,
        json_path,
        tokenizer,
        max_length=MAX_LENGTH,
        input_field="input",
        output_field="target",
        system_field="system"
    ):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.input_field = input_field
        self.output_field = output_field
        self.system_field = system_field
        self.has_chat_template = hasattr(tokenizer, 'apply_chat_template')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item[self.input_field]
        answer = item[self.output_field]
        system_prompt = item.get(self.system_field, "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        messages.append({"role": "assistant", "content": answer})

        if self.has_chat_template:
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            prompt_messages = []
            if system_prompt:
                prompt_messages.append({"role": "system", "content": system_prompt})
            prompt_messages.append({"role": "user", "content": question})
            prompt_text = self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            bos = self.tokenizer.bos_token or ""
            full_text = bos
            if system_prompt:
                full_text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            full_text += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n{answer}<|im_end|>\n"

            prompt_text = bos
            if system_prompt:
                prompt_text += f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            prompt_text += f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        full_enc = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )
        prompt_enc = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
            add_special_tokens=False
        )

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc["attention_mask"]
        prompt_len = len(prompt_enc["input_ids"])

        labels = input_ids.copy()
        labels[:prompt_len] = [-100] * prompt_len
        if self.tokenizer.pad_token_id is not None:
            labels = [-100 if x == self.tokenizer.pad_token_id else x for x in labels]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def collate_fn(batch, tokenizer):
    input_ids = [item["input_ids"] for item in batch]
    attention_mask = [item["attention_mask"] for item in batch]
    labels = [item["labels"] for item in batch]

    max_len = max(len(ids) for ids in input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []

    for ids, mask, lbls in zip(input_ids, attention_mask, labels):
        pad_len = max_len - len(ids)
        padded_input_ids.append(ids + [pad_token_id] * pad_len)
        padded_attention_mask.append(mask + [0] * pad_len)
        padded_labels.append(lbls + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }

# ==============================
# Загрузка модели и токенизатора
# ==============================
print("Загружаю токенизатор...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Загружаю базовую модель...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)

# Отключаем кеш и включаем checkpointing для экономии памяти
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ==============================
# Конфигурация LoRA
# ==============================
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=TARGET_MODULES,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==============================
# Датасеты и DataLoader'ы
# ==============================
print("Создаю датасеты...")
train_dataset = UniversalDialogDataset(TRAIN_JSON_PATH, tokenizer, max_length=MAX_LENGTH)
valid_dataset = UniversalDialogDataset(VALID_JSON_PATH, tokenizer, max_length=MAX_LENGTH)

collate = partial(collate_fn, tokenizer=tokenizer)

train_loader = DataLoader(
    train_dataset,
    batch_size=PHYSICAL_BATCH_SIZE,
    shuffle=True,
    collate_fn=collate
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=PHYSICAL_BATCH_SIZE,
    shuffle=False,
    collate_fn=collate
)

# ==============================
# Оптимизатор
# ==============================
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE
)

# ==============================
# Цикл обучения
# ==============================
train_losses = []
valid_losses = []

for ep in range(EPOCHS):
    model.train()
    TrainLoss = 0.0
    step_in_accum = 0
    num_batches = len(train_loader)

    pbar = tqdm(enumerate(train_loader), total=num_batches, desc=f"Epoch {ep+1}/{EPOCHS}")

    for batch_idx, batch in pbar:
        outputs = model(
            input_ids=batch["input_ids"].to(model.device),
            attention_mask=batch["attention_mask"].to(model.device),
            labels=batch["labels"].to(model.device)
        )

        loss = outputs.loss
        TrainLoss += loss.item()

        (loss / GRAD_ACCUM_STEPS).backward()
        step_in_accum += 1

        if step_in_accum == GRAD_ACCUM_STEPS or (batch_idx + 1) == num_batches:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step_in_accum = 0

        current_avg = TrainLoss / (batch_idx + 1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{current_avg:.4f}"})

    TrainLoss /= num_batches
    train_losses.append(TrainLoss)
    print(f"\nEpoch {ep+1} - train loss: {TrainLoss:.4f}")

    # Сохраняем адаптер после каждой эпохи
    model.save_pretrained(os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{ep+1}"))

    # Валидация
    model.eval()
    Valid_loss = 0.0
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(
                input_ids=batch["input_ids"].to(model.device),
                attention_mask=batch["attention_mask"].to(model.device),
                labels=batch["labels"].to(model.device)
            )
            Valid_loss += outputs.loss.item()

    Valid_loss /= len(valid_loader)
    valid_losses.append(Valid_loss)
    print(f"Epoch {ep+1} - valid loss: {Valid_loss:.4f}\n")

# ==============================
# Сохранение финального адаптера
# ==============================
model.save_pretrained(OUTPUT_DIR)
print(f"Финальный адаптер сохранён в {OUTPUT_DIR}")

# ==============================
# График лоссов
# ==============================
plt.figure(figsize=(10, 6))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', label='Train Loss')
plt.plot(range(1, EPOCHS+1), valid_losses, marker='s', label='Valid Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('LoRA Fine-tuning: Train vs Valid Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(range(1, EPOCHS+1))
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loss_curves.png'), dpi=150)
print("График сохранён как loss_curves.png")
plt.show()