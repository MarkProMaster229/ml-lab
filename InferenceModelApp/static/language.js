// DOM элементы
const modelSelect = document.getElementById('modelType');
const promptInput = document.getElementById('promptInput');
const generateBtn = document.getElementById('generateBtn');
const resultDiv = document.getElementById('result');
const resultContent = document.getElementById('resultContent');
const loadingDiv = document.getElementById('loading');

// Генерация по кнопке
generateBtn.addEventListener('click', () => {
    const prompt = promptInput.value.trim();
    if (!prompt) {
        showError('Введите промпт перед генерацией');
        return;
    }
    generateText(prompt);
});

// Генерация по Ctrl+Enter
promptInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        generateBtn.click();
    }
});

async function generateText(prompt) {
    // Показываем загрузку
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    const requestData = {
        prompt: prompt,
        model_type: parseInt(modelSelect.value)
    };
    
    try {
        const response = await fetch('/predict_language', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data.generated_text);
        } else {
            showError(data.error);
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Ошибка соединения с сервером');
    } finally {
        loadingDiv.classList.add('hidden');
    }
}

function showResult(text) {
    // Экранируем HTML специальные символы
    const escapedText = escapeHtml(text);
    // Сохраняем переносы строк
    const formattedText = escapedText.replace(/\n/g, '<br>');
    
    resultContent.innerHTML = `
        <div class="result-card">
            <div class="generated-text">${formattedText}</div>
            <div class="result-meta">
                <span class="copy-btn" onclick="copyToClipboard()">Копировать</span>
            </div>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function showError(error) {
    resultContent.innerHTML = `
        <div class="error-card">
            <p>Ошибка: ${escapeHtml(error)}</p>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function copyToClipboard() {
    const text = resultContent.querySelector('.generated-text').innerText;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.querySelector('.copy-btn');
        const originalText = btn.innerHTML;
        btn.innerHTML = 'Скопировано!';
        setTimeout(() => {
            btn.innerHTML = originalText;
        }, 2000);
    });
}