# data.py
import json
import torch
from torch.utils.data import Dataset, DataLoader


class PromptCompletionDataset(Dataset):
    """
    Датасет для задачи «продолжи текст».
    
    Формат: [{"input": "...", "target": "..."}, ...]
    
    Логика:
      - Примеры длиннее seq_len — ВЫБРАСЫВАЮТСЯ (не режутся!)
      - loss считается ТОЛЬКО на target (input маскируется -100)
    """
    def __init__(self, data, tokenizer, seq_len, add_eos=True):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.add_eos = add_eos
        self.samples = []
        
        n_empty = 0
        n_too_long = 0

        for item in data:
            inp = item["input"].strip()
            tgt = item["target"].strip()
            if not inp or not tgt:
                n_empty += 1
                continue

            inp_ids = tokenizer.encode(inp, add_special_tokens=False)
            tgt_ids = tokenizer.encode(tgt, add_special_tokens=False)

            if add_eos:
                tgt_ids = tgt_ids + [tokenizer.eos_token_id]

            # Если длиннее seq_len — ВЫБРАСЫВАЕМ
            if len(inp_ids) + len(tgt_ids) > seq_len:
                n_too_long += 1
                continue

            tokens = inp_ids + tgt_ids
            labels = [-100] * len(inp_ids) + tgt_ids

            self.samples.append((tokens, labels))

        kept = len(self.samples)
        total = kept + n_empty + n_too_long
        print(f"[dataset] kept {kept}/{total} "
              f"(empty {n_empty}, too_long {n_too_long}, "
              f"dropped {100*(n_empty+n_too_long)/max(total,1):.1f}%)")
        
        if kept == 0:
            raise ValueError(f"Датасет пустой! Все {len(data)} примеров выброшены.")

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
    def __init__(
        self,
        data,
        tokenizer,
        seq_len=1024,
        batch_size=8,
        val_ratio=0.1,
        num_workers=2,
        seed=42,
        add_eos=True,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers

        rng = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(data), generator=rng).tolist()
        n_val = max(1, int(len(data) * val_ratio))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]

        print("Train split:")
        self.train_ds = PromptCompletionDataset(train_data, tokenizer, seq_len, add_eos)
        print("Val split:")
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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)