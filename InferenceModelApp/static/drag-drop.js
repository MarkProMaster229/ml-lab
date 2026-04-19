// DOM элементы
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileSelectBtn = document.getElementById('fileSelectBtn');
const modelSelect = document.getElementById('modelType');
const resultDiv = document.getElementById('result');
const resultContent = document.getElementById('resultContent');
const loadingDiv = document.getElementById('loading');
const testFolderBtn = document.getElementById('testFolderBtn');
const testResultsDiv = document.getElementById('testResults');

if (fileSelectBtn) {
    fileSelectBtn.addEventListener('click', () => {
        fileInput.click();
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            processImage(e.target.files[0]);
        }
    });
}
if (dropZone) {
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            processImage(file);
        } else {
            showError('Пожалуйста, перетащите изображение (PNG, JPG, JPEG)');
        }
    });
}

if (testFolderBtn) {
    testFolderBtn.addEventListener('click', async () => {
        const modelType = modelSelect ? modelSelect.value : 1;
        
        testFolderBtn.disabled = true;
        testFolderBtn.textContent = 'Тестирование...';
        
        const formData = new FormData();
        formData.append('model_type', modelType);
        
        try {
            const response = await fetch('/test_folder', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (data.success) {
                showTestResults(data.results);
            } else {
                showErrorInTest(data.error);
            }
        } catch (error) {
            showErrorInTest('Ошибка соединения с сервером');
        } finally {
            testFolderBtn.disabled = false;
            testFolderBtn.textContent = 'Запустить тест папки';
        }
    });
}

async function processImage(file) {
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    const formData = new FormData();
    formData.append('image', file);
    formData.append('model_type', modelSelect ? modelSelect.value : 1);
    
    try {
        const response = await fetch('/predict_cnn', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data);
        } else {
            showError(data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Ошибка соединения с сервером. Убедитесь, что сервер запущен.');
    } finally {
        loadingDiv.classList.add('hidden');
    }
}

function showResult(data) {
    const confidencePercent = (data.confidence * 100).toFixed(2);
    
    resultContent.innerHTML = `
        <div class="result-card">
            <p style="margin-bottom: 15px;">
                <strong>Предсказанная буква:</strong>
            </p>
            <div class="letter" style="font-size: 64px; margin-bottom: 15px;">
                ${data.label}
            </div>
            <p style="margin-top: 15px;">
                <strong>Уверенность:</strong>
                <span class="confidence">${confidencePercent}%</span>
            </p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: ${confidencePercent}%">
                    ${confidencePercent}%
                </div>
            </div>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function showError(error) {
    resultContent.innerHTML = `
        <div class="error-card">
            <p style="font-size: 48px; margin-bottom: 10px;">нет</p>
            <p><strong>Ошибка:</strong></p>
            <p style="color: #e53e3e;">${error}</p>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function showTestResults(results) {
    if (!testResultsDiv) return;
    
    if (!results || results.length === 0) {
        testResultsDiv.innerHTML = '<p>Изображений не найдено в папке myLetterTest</p>';
    } else {
        let html = '<h4>Результаты теста:</h4>';
        results.forEach(r => {
            const confPercent = (r.confidence * 100).toFixed(1);
            html += `
                <div class="test-result-item">
                     ${r.file} → <strong>${r.label}</strong> (${confPercent}%)
                </div>
            `;
        });
        html += `<p style="margin-top: 10px; font-weight: bold;"> Обработано: ${results.length} изображений</p>`;
        testResultsDiv.innerHTML = html;
    }
    testResultsDiv.classList.remove('hidden');
}

function showErrorInTest(error) {
    if (testResultsDiv) {
        testResultsDiv.innerHTML = `<p style="color: #e53e3e;"> ${error}</p>`;
        testResultsDiv.classList.remove('hidden');
    }
}


console.log('Drag & Drop готов к работе!');