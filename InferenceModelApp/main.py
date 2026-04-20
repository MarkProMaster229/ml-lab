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

from managerModel import Manager
import kagglehub

from pathlib import Path

managerForModel = Manager()


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