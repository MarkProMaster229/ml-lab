from huggingface_hub import snapshot_download
import os

repo_id = "MarkProMaster229/experimental_models"


local_dir = "/home/chelovek/allMyWork"

print(f"Начинаю загрузку репозитория {repo_id}...")
print(f"Это может занять много времени, так как размер репозитория около 18 ГБ.")
print(f"Файлы будут сохранены в: {os.path.abspath(local_dir)}")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print("Загрузка успешно завершена!")
except Exception as e:
    print(f"Произошла ошибка: {e}")