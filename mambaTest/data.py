# data.py
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class PromptCompletionDataset(Dataset):
    """
    Датасет для задачи «продолжи текст».
    
    Формат входа: список словарей [{"input": "...", "target": "..."}, ...]
    
    Логика:
      - tokenizer(input) + tokenizer(target) → одна последовательность
      - labels = [-100] * len(input_tokens) + target_tokens
        (loss считается ТОЛЬКО на target)
      - паддинг до seq_len
    """
    def __init__(self, data, tokenizer, seq_len, add_eos=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        self.samples = []

        for item in data:
            inp = item["input"].strip()
            tgt = item["target"].strip()
            if not inp or not tgt:
                continue  # пропускаем пустые

            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)

            if add_eos:
                tgt_ids = tgt_ids + [tokenizer.eos_token_id]

            # Вся последовательность: input + target
            tokens = inp_ids + tgt_ids
            # Метки: -100 на input, реальные id на target
            labels = [-100] * len(inp_ids) + tgt_ids

            # Обрезаем до seq_len + 1 (нужен сдвиг на 1)
            tokens = tokens[: seq_len + 1]
            labels = labels[: seq_len + 1]

            # Если target обрезался полностью — пропускаем
            if all(l == -100 for l in labels[1:]):
                continue

            self.samples.append((tokens, labels))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens, labels = self.samples[idx]

        # Сдвиг: input_ids[:-1], labels[1:]
        input_ids = tokens[:-1]
        target_ids = labels[1:]

        # Паддинг до seq_len
        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * pad_len
            target_ids = target_ids + [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(target_ids, dtype=torch.long),
        }


class DataModule:
    """
    Управляет train/val сплитом и создаёт DataLoader'ы.
    
    Использование:
        dm = DataModule(data, tokenizer, seq_len=512, batch_size=4, val_ratio=0.1)
        train_loader = dm.train_dataloader()
        val_loader = dm.val_dataloader()
    """
    def __init__(
        self,
        data,
        tokenizer,
        seq_len=512,
        batch_size=4,
        val_ratio=0.1,
        num_workers=2,
        seed=42,
        add_eos=True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Перемешиваем и делим
        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(data), generator=rng).tolist()
        n_val = max(1, int(len(data) * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        self.train_ds = PromptCompletionDataset(train_data, tokenizer, seq_len, add_eos)
        self.val_ds = PromptCompletionDataset(val_data, tokenizer, seq_len, add_eos)

        print(f"Train: {len(self.train_ds)} примеров | Val: {len(self.val_ds)} примеров")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )


def load_json_data(path):
    """Загружает JSON-файл со списком dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)