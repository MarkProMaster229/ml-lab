from decoderOnly import TransformerRun
from transformers import AutoTokenizer
import torch

class ChatBot:
    def __init__(self, model_path="trained_model"):
        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Создаем модель с параметрами токенизатора
        self.model = TransformerRun(
            vocabSize=len(self.tokenizer), 
            maxLong=256,
            sizeVector=128,
            block=2
            )

        
        # Загружаем веса модели
        self.model.load_state_dict(
            torch.load(f"{model_path}/model_weights.pth", map_location='cpu', weights_only=True)
        )
        
        # Настраиваем устройство
        self.device = torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate(self, prompt, max_length=50, temperature=0.1, top_k=1):
        # Токенизируем промпт
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generated_ids = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_length):
                # Прямой проход модели
                outputs = self.model(generated_ids)
                logits = outputs[0, -1, :] / temperature  # учитываем температуру

                # Top-k sampling
                if top_k > 0:
                    topk_values, topk_indices = torch.topk(logits, top_k)
                    probs = torch.zeros_like(logits).scatter(0, topk_indices, torch.softmax(topk_values, dim=-1))
                else:
                    probs = torch.softmax(logits, dim=-1)

                # Выбираем следующий токен
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Добавляем к сгенерированной последовательности
                generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                
                # Останавливаемся на EOS или PAD
                if next_token.item() in [self.tokenizer.eos_token_id, self.tokenizer.pad_token_id]:
                    break

        # Декодируем обратно в текст
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

if __name__ == "__main__":
    bot = ChatBot("/home/chelovek/Музыка/epoch_35")
    
    print("Бот готов! Напиши 'exit' для выхода.")
    
    while True:
        prompt = input("Ты: ")
        if prompt.lower() in ["exit", "quit"]:
            break
        response = bot.generate(prompt, max_length=100, temperature=0.5, top_k=50)
        print(f"Бот: {response}")
