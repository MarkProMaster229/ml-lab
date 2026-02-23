# r_alternative.py - с использованием обычного Trainer
#Qwen2.5-1.5B-Instruct
import torch
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_from_disk
import os
os.environ['TRANSFORMERS_AWQ_BACKEND'] = 'autoawq'

BASE_MODEL_PATH = "/home/chelovek/work/model4b_8bit_config"
DATASET_PATH = "/home/chelovek/work/lora_project/datasets/my_dataset/"
OUTPUT_DIR = "/home/chelovek/work/lora_project/models/lora_adapters23"

# config LoRA
lora_config = LoraConfig(
    #толщина чем больше тем больше влияет на конечные веса
    r=4,
    #СИЛА ВЛИЯНИЯ
    lora_alpha=6,
    # слои тут надо разжевать мне qkw это понятно это матрицы внимания а че ткое "gate_proj", "up_proj", "down_proj" я ваще не ебу ааааа 
    #я пон эт gate_proj решает че обрабатывать Up - "расширяет представление" own_proj - сжимает, ну типичный трансформер))) 
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #ну дропаут
    lora_dropout=0.05,
    # типо не трогаем биас
    bias="none",
    #казуальная маска по типу задачи
    task_type=TaskType.CAUSAL_LM,
)


print("this model download")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    trust_remote_code=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# using Lora
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )

print("dowload model and tokeni.")
dataset = load_from_disk(DATASET_PATH)

# tokenization
# create map token
tokenized_train = dataset["train"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
#test kit dataset
tokenized_test = dataset["test"].map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["test"].column_names
)

# this config lern model
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    #в первый раз пробую этот хак, чуть позже посмотрю лучше или как
    gradient_accumulation_steps=12,
    #адаптивно поддрачивает lr  ПЛАВНЫЙ СТАРТ Первые 10 шагов lr от 0 до 2e-4
    warmup_steps=10,
    logging_steps=10,
    save_steps=10,
    eval_strategy="steps",
    eval_steps=10,
    learning_rate=2e-4,
    fp16=True,
    optim="adamw_torch",
    #тоже впервые пробую - Вместо хранения ВСЕХ промежуточных значений для обратного распространения Пересчитывает их на лету когда нужно
    gradient_checkpointing=True,
    report_to="none",
    save_total_limit=2,
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 8. СОЗДАНИЕ ТРЕНЕРА
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
)

print("=" * 60)
print("🎯 СТАРТ ОБУЧЕНИЯ LoRA")
print(f"📊 Размер тренировочного датасета: {len(tokenized_train)}")
print(f"📊 Размер тестового датасета: {len(tokenized_test)}")
print(f"⚙️  Параметры обучения:")
print(f"   - Эпохи: {training_args.num_train_epochs}")
print(f"   - Learning rate: {training_args.learning_rate}")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Длина последовательности: 256 токенов")
print("=" * 60)


# 9. ОБУЧЕНИЕ
print("Старт обучения...")
trainer.train()


# 10. СОХРАНЕНИЕ
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)


print(f"Обучение завершено! Сохранено в: {OUTPUT_DIR}")