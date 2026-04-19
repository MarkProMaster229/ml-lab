from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import gc
from flask import Flask, request, jsonify, render_template_string, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename

from engineModel.engAutoreg import EngineAutoreg

from engineModel.baseBert import BaseBert
from engineModel.distilBert import DistilBert
from engineModel.engineRoBert import EngineRoBert
from engineModel.engineClassificationSmall import EngineTransformerClassifier

from engineModel.engineRESNET18 import EngineRESNET18
from engineModel.engineMobileNetV2 import EngineMobileNetV2
from engineModel.engineSE_ResNet18 import EngineSEResNetClassifier

from engineModel.engineU_net import EngineU_net
from engineModel.engineAttentionUNet import EngineAttentionUNet
from engineModel.engineUNet3D import EngineUNet3D
from engineModel.engineAttentionUNet3D import EngineAttentionUNet3D

import kagglehub
#this collector
from pathlib import Path
DATASET_NAME = "andrewmvd/pulmonary-embolism-in-ct-images"

def get_dataset_path():
    path = kagglehub.dataset_download(DATASET_NAME)
    return Path(path) / "FUMPE"

dataset_path = get_dataset_path()

from pathlib import Path
import os

def get_images_from_folder(folder_path):
    folder = Path(folder_path)
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for image_path in folder.iterdir():
        if image_path.suffix.lower() in extensions:
            yield str(image_path)


#image_folder = "myLetterTest"

#for image_path in get_images_from_folder(image_folder):
#    result = obj.ThisControllerImageLetterModel(image_path, MyMagicObject=1)

#y.evaluate_invariance2(
#    dcm_dir=dataset_path / "PAT034",
#    mat_path=dataset_path / "PAT034.mat",
#    threshold=0.5
#)

#main
class Manager:
    def MyCollector(self, model):
        if hasattr(model, 'model'):
            model.model.cpu()
            del model.model
        if hasattr(model, 'tokenizer'):
            del model.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        del model
        gc.collect()

        gc.collect()

    #this Load-Use-Unload
    
    def ThisControllerLanguageModel(self, promt, MyMagicObject):
        # my magic logic
        # archAutoRegr
        if MyMagicObject == 1:
            model = EngineAutoreg()
            result = model.generate(promt)
            self.MyCollector(model)
            print(result)
            return result
        #bertBaseVanila 
        elif MyMagicObject == 2:
            model = BaseBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #distilBert 
        elif MyMagicObject == 3:
            model = DistilBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #Robert
        elif MyMagicObject == 4:
            model = EngineRoBert()
            result = model.predict(promt)
            self.MyCollector(model)
            print(result)
            return result
        #ClassificationSmall
        elif MyMagicObject == 5:
            model = EngineTransformerClassifier()
            result = model.predict()
            self.MyCollector(model)
            print(result)
            return result

    def ThisControllerImageLetterModel(self, image, MyMagicObject):
        #ResNet18(Vanilla)
        if MyMagicObject == 1:
            model = EngineRESNET18()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
        #MobileNetV2
        elif MyMagicObject == 2:
            model = EngineMobileNetV2()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
        #SE-ResNet18
        elif MyMagicObject == 3:
            model = EngineSEResNetClassifier()
            result = model.predict(image)
            self.MyCollector(model)
            print(result)
            return result
#y.evaluate_invariance2(
#    dcm_dir=dataset_path / "PAT034",
#    mat_path=dataset_path / "PAT034.mat",
#    threshold=0.5
#)
        #end embol.
    def ThisControllerImageEmbol(self, MyMagicObject):
        dataset_path = get_dataset_path()
    
        dcm_dir = dataset_path / "CT_scans" / "PAT034"
        mat_path = dataset_path / "GroundTruth" / "PAT034.mat"
        
        #U_net2d
        if MyMagicObject == 1:
            model = EngineU_net()
            model.demo()
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            self.MyCollector(model)
        #U-Net attention2d
        elif MyMagicObject == 2:
            model = EngineAttentionUNet()
            import os
            print(f"dataset_path = {dataset_path}")
            print(f"Содержимое: {os.listdir(dataset_path)}")
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)
        #3D U-Net witch VGG encoder
        elif MyMagicObject == 3:
            import os
            model = EngineUNet3D()
            print(f"dataset_path = {dataset_path}")
            print(f"Содержимое: {os.listdir(dataset_path)}")
            model.evaluate_invariance(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)
        #3D Attention U-Net
        elif MyMagicObject == 4:
            model = EngineAttentionUNet3D()
            model.evaluate_invariance2(
                dcm_dir=dcm_dir,
                mat_path=mat_path,
                threshold=0.5
            )
            model.demo()
            self.MyCollector(model)

            
#deletMe If you using refactoring!
    def export_embol_visualizations(self, MyMagicObject, dcm_dir, mat_path, output_root):
        """
        Экспортирует все визуализации для указанной модели.
        
        Args:
            MyMagicObject: номер модели (1-4)
            dcm_dir: путь к DICOM файлам
            mat_path: путь к .mat файлу
            output_root: корневая папка для сохранения
        """
        output_path = Path(output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Выбираем модель
        if MyMagicObject == 1:
            model = EngineU_net()
            model_name = "unet_2d"
        elif MyMagicObject == 2:
            model = EngineAttentionUNet()
            model_name = "attention_unet_2d"
        elif MyMagicObject == 3:
            model = EngineUNet3D()
            model_name = "unet_3d"
        elif MyMagicObject == 4:
            model = EngineAttentionUNet3D()
            model_name = "attention_unet_3d"
        else:
            raise ValueError(f"Unknown model: {MyMagicObject}")
        
        print(f"\n🚀 Экспорт для модели: {model_name}")
        
        # Вызываем метод экспорта
        model.export_all_transformations(
            dcm_dir=str(dcm_dir),
            mat_path=str(mat_path),
            output_dir=output_path / model_name
        )
        
        # Чистим память
        self.MyCollector(model)
#saveme!
managerForModel = Manager()

#saveme!
def main():
    # Пути к данным
    dcm_dir = dataset_path / "CT_scans" / "PAT034"
    mat_path = dataset_path / "GroundTruth" / "PAT034.mat"
    output_root = Path("static/precomputed/PAT034")
    
    # Проверяем, что данные существуют
    if not dcm_dir.exists():
        print(f"❌ Ошибка: DICOM папка не найдена: {dcm_dir}")
        return
    
    if not mat_path.exists():
        print(f"⚠️ Предупреждение: GT файл не найден: {mat_path}")
        print("   Будет сохранено только КТ и предсказания")
    
    # Создаём менеджера
    manager = Manager()
    
    # Список моделей для экспорта
    models = [
        (1, "2D U-Net"),
        (2, "2D Attention U-Net"),
        (3, "3D U-Net"),
        (4, "3D Attention U-Net"),
    ]
    
    print("=" * 60)
    print("🚀 НАЧАЛО ГЕНЕРАЦИИ ВИЗУАЛИЗАЦИЙ")
    print("=" * 60)
    
    for model_id, model_name in models:
        print(f"\n📦 Обработка: {model_name}")
        manager.export_embol_visualizations(
            MyMagicObject=model_id,
            dcm_dir=dcm_dir,
            mat_path=mat_path,
            output_root=output_root
        )
    
    print("\n" + "=" * 60)
    print(f"✅ ГЕНЕРАЦИЯ ЗАВЕРШЕНА!")
    print(f"📁 Результаты сохранены в: {output_root.absolute()}")
    print("=" * 60)
#stop refactoring ----------------------
#if __name__ == "__main__":
#    main()

app = Flask(__name__)
CORS(app)

# Конфигурация
UPLOAD_FOLDER = Path('/tmp/inference_uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True, parents=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB макс

# Разрешённые расширения
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Главная страница с CNN моделями"""
    return render_template('cnn_models.html')

@app.route('/predict_cnn', methods=['POST'])
def predict_cnn():
    """Эндпоинт для предсказания по изображению"""
    try:
        # Проверяем, есть ли файл
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'Файл не загружен'})
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'Файл не выбран'})
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Неподдерживаемый формат. Используйте: png, jpg, jpeg, bmp, tiff'})
        
        # Сохраняем временно
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(filepath)
        
        # Получаем тип модели
        model_type = int(request.form.get('model_type', 1))
        
        # Вызываем твою state-машину
        label, confidence = managerForModel.ThisControllerImageLetterModel(
            str(filepath), 
            model_type
        )
        
        # Удаляем временный файл (комментируй, если нужно сохранять)
        try:
            filepath.unlink()
        except:
            pass
        
        return jsonify({
            'success': True,
            'label': label,
            'confidence': confidence
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

@app.route('/test_folder', methods=['POST'])
def test_folder():
    """Тестовый эндпоинт для прогона папки myLetterTest"""
    try:
        model_type = int(request.form.get('model_type', 1))
        test_folder_path = Path("myLetterTest")
        
        if not test_folder_path.exists():
            return jsonify({'success': False, 'error': f'Папка {test_folder_path} не найдена'})
        
        results = []
        for img_path in test_folder_path.iterdir():
            if img_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp']:
                label, confidence = managerForModel.ThisControllerImageLetterModel(
                    str(img_path), model_type
                )
                results.append({
                    'file': img_path.name,
                    'label': label,
                    'confidence': confidence
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(results)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    
@app.route('/language')
def language_models():
    """Страница для авторегрессивных языковых моделей"""
    return render_template('language_models.html')

@app.route('/predict_language', methods=['POST'])
def predict_language():
    """Эндпоинт для генерации текста"""
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')
        model_type = int(data.get('model_type', 1))
        
        if not prompt.strip():
            return jsonify({'success': False, 'error': 'Промпт не может быть пустым'})
        
        # Вызываем твою state-машину
        result = managerForModel.ThisControllerLanguageModel(prompt, model_type)
        
        # result — это то, что возвращает твой метод (скорее всего строка)
        return jsonify({
            'success': True,
            'generated_text': result
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})
    
    from huggingface_hub import snapshot_download
from pathlib import Path
import shutil

# Глобальная папка для кеша прекомпьютед картинок
PRECOMPUTED_CACHE = Path("/tmp/inference_precomputed")
PRECOMPUTED_REPO = "MarkProMaster229/experimental_models"  # твой репозиторий

def ensure_precomputed_available():
    """Проверяет, есть ли локально precomputed, если нет — скачивает с Hugging Face"""
    local_precomputed = PRECOMPUTED_CACHE / "precomputed"
    
    # Если уже есть — выходим
    if local_precomputed.exists() and any(local_precomputed.iterdir()):
        print(f"✅ Precomputed уже есть: {local_precomputed}")
        return local_precomputed
    
    print(f"📥 Скачиваю precomputed с Hugging Face...")
    print(f"   Репозиторий: {PRECOMPUTED_REPO}")
    print(f"   Папка: precomputed/")
    from huggingface_hub import snapshot_download
    try:
        # Скачиваем только папку precomputed
        snapshot_download(
            repo_id=PRECOMPUTED_REPO,
            allow_patterns=["precomputed/*"],  # только precomputed
            local_dir=PRECOMPUTED_CACHE,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"✅ Скачивание завершено: {local_precomputed}")
        return local_precomputed
    except Exception as e:
        print(f"❌ Ошибка скачивания: {e}")
        # Если скачать не удалось — пробуем использовать локальную папку (для разработки)
        local_fallback = Path("static/precomputed")
        if local_fallback.exists():
            print(f"⚠️ Использую локальную папку: {local_fallback}")
            return local_fallback
        raise

@app.route('/embolism')
def embolism_models():
    """Страница для визуализации эмболии (сегментация)"""
    return render_template('embolism.html')

@app.route('/api/embolism/images')
def get_embolism_images():
    """
    Возвращает список доступных моделей, трансформаций и срезов.
    Фронт использует этот эндпоинт для построения UI.
    """
    precomputed_path = ensure_precomputed_available()
    
    # Структура: precomputed/PAT034/{model_name}/{transform}/
    pat034_path = precomputed_path / "PAT034"
    
    if not pat034_path.exists():
        return jsonify({'success': False, 'error': 'Precomputed data not found'})
    
    models = []
    for model_dir in pat034_path.iterdir():
        if model_dir.is_dir():
            transforms = []
            for transform_dir in model_dir.iterdir():
                if transform_dir.is_dir():
                    # Считаем количество срезов по файлам
                    slices = set()
                    for f in transform_dir.glob("slice_*_ct.png"):
                        # Извлекаем номер среза из имени slice_X_ct.png
                        import re
                        match = re.search(r'slice_(\d+)_ct\.png', f.name)
                        if match:
                            slices.add(int(match.group(1)))
                    
                    transforms.append({
                        'name': transform_dir.name,
                        'slices': sorted(list(slices))
                    })
            
            models.append({
                'name': model_dir.name,
                'transforms': transforms
            })
    
    return jsonify({
        'success': True,
        'models': models,
        'base_path': '/embolism/static'
    })

@app.route('/embolism/static/<path:filename>')
def embolism_static(filename):
    """Отдаёт precomputed картинки как статику"""
    precomputed_path = ensure_precomputed_available()
    file_path = precomputed_path / filename
    
    # Безопасность: проверяем, что путь внутри precomputed
    try:
        file_path.resolve().relative_to(precomputed_path.resolve())
    except ValueError:
        return jsonify({'error': 'Invalid path'}), 403
    
    if not file_path.exists():
        return jsonify({'error': 'Image not found'}), 404
    
    from flask import send_file
    return send_file(file_path, mimetype='image/png')

if __name__ == '__main__':
    print(" Запуск сервера...")
    print(f" Временные файлы: {UPLOAD_FOLDER}")
    app.run(debug=True, host='0.0.0.0', port=5000)