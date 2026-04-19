from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
from pathlib import Path
from werkzeug.utils import secure_filename
import torch
import gc

# Импорт твоего Manager (предполагаю, что он в том же файле или импортится)
# Если Manager в другом файле, раскомментируй:
# from your_manager_file import Manager

# ========== ВРЕМЕННО: твой Manager (для целостности примера) ==========
# Если у тебя уже есть Manager в другом файле, ЗАКОММЕНТИРУЙ этот блок
class Manager:
    def MyCollector(self, model):
        if hasattr(model, 'model'):
            if hasattr(model.model, 'cpu'):
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
    
    def ThisControllerImageLetterModel(self, image, MyMagicObject):
        # Здесь твои реальные Engine'ы
        # Сейчас заглушка для теста
        print(f"Обработка изображения: {image}")
        print(f"Тип модели: {MyMagicObject}")
        
        # Заглушка результата (замени на реальный код)
        if MyMagicObject == 1:
            label = "А"
            confidence = 0.95
        elif MyMagicObject == 2:
            label = "Б"
            confidence = 0.87
        else:
            label = "В"
            confidence = 0.92
        
        return label, confidence

# Создаём глобальный экземпляр Manager
managerForModel = Manager()
# =====================================================================

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

if __name__ == '__main__':
    print("🚀 Запуск сервера...")
    print(f"📁 Временные файлы: {UPLOAD_FOLDER}")
    print("🌐 Открой в браузере: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)