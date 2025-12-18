from encoderOInly import TransformerRun  # ← это МОДЕЛЬ
from tokenizer import TokenizerForClassification  # ← это ТОКЕНИЗАТОР (добавь!)
import torch
import torch.optim as optim
import torch.nn as nn
#partially generated ai
class ClassifierModel():
    def __init__(self, sizeVector=512, num_layers=16, maxLong=100):
        # 1. ТОКЕНИЗАТОР - отдельный класс
        self.tokenizer = TokenizerForClassification()  # ← ТОКЕНИЗАТОР!
        
        self.vocabSize = self.tokenizer.get_vocab_size()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sizeVector = sizeVector
        self.num_layers = num_layers
        self.maxLong = maxLong

        # 2. МОДЕЛЬ - отдельно
        self.model = TransformerRun(
            vocabSize=self.vocabSize,
            maxLong=self.maxLong,
            sizeVector=sizeVector,
            block=num_layers
        ).to(self.device)
        
        print(f"Классификация на")
        print(f"Размер словаря: {self.vocabSize}")
        print(f"Макс. длина: {maxLong}")
        print(f"Устройство: {self.device}")

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_model(self):
        #no using ai!
        dataloader = self.tokenizer.dataloader()
        numEpoch = 120

        for epoch in range(numEpoch):
            print(f"Эпоха {epoch+1}/{numEpoch}")
            self.model.train()
            
            if (epoch + 1) % 120 == 0:
                self.save_model(f"classifier_epoch{epoch+1}")

            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(input_ids)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    with torch.no_grad():
                        predictions = torch.argmax(outputs, dim=1)
                        accuracy = (predictions == labels).float().mean()
                    
                    #print(f"  Batch {batch_idx}/{len(dataloader)} - "
                          #f"Loss: {loss.item():.4f}, Acc: {accuracy:.2f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Эпоха {epoch+1} завершена. Средний loss: {avg_loss:.4f}")
    
            # 11. Можно добавить валидацию
            # self.evaluate()
    
    # 12. Метод для валидации (опционально)
    def evaluate(self):
        self.model.eval()
        dataloader = self.tokenizer.dataloader()
        
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids)
                predictions = torch.argmax(outputs, dim=1)
                
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        print(f"Точность на всём датасете: {accuracy:.2%}")
        return accuracy
    #ai
    # 13. Метод для предсказания
    def predict(self, text):
        self.model.eval()
        with torch.no_grad():
            # Токенизация текста
            encoded = self.tokenizer.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=self.maxLong,
                return_tensors="pt"
            )
            
            input_ids = encoded["input_ids"].to(self.device)
            outputs = self.model(input_ids)
            predicted_class = torch.argmax(outputs, dim=1).item()
            
            # Преобразуем обратно в строку
            label_name = self.tokenizer.get_label_name(predicted_class)
            
            return predicted_class, label_name, outputs.softmax(dim=1)
    #not ai 
    def save_model(self, path="my_classifier"):
        import os
        os.makedirs(path, exist_ok=True)
        
        # Сохраняем веса модели
        torch.save(self.model.state_dict(), f"{path}/model_weights.pth")
        
        # Конфигурация
        config = {
            'vocabSize': self.vocabSize,
            'maxLong': self.maxLong,
            'sizeVector': self.sizeVector,
            'numLayers': self.num_layers,
            'device': str(self.device),
            'num_classes': 3  # ← добавил
        }
        torch.save(config, f"{path}/config.pth")
        
        # Токенизатор
        self.tokenizer.tokenizer.save_pretrained(path)
        
        # Optimizer
        torch.save(self.optimizer.state_dict(), f"{path}/optimizer.pth")
        
        # Label mapping
        torch.save(self.tokenizer.label_map, f"{path}/label_map.pth")
        
        print(f"Модель сохранена в: {path}")

# 14. Запуск
if __name__ == "__main__":
    # Проверяем, что maxLong совпадает с токенизатором (100)
    model = ClassifierModel(sizeVector=512, num_layers=12, maxLong=100)
    model.train_model()
    model.save_model("trained_classifier")