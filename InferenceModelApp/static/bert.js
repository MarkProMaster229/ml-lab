const modelSelect = document.getElementById('modelType');
const textInput = document.getElementById('textInput');
const predictBtn = document.getElementById('predictBtn');
const resultDiv = document.getElementById('result');
const resultContent = document.getElementById('resultContent');
const loadingDiv = document.getElementById('loading');

predictBtn.addEventListener('click', () => {
    const text = textInput.value.trim();
    if (!text) {
        showError('Введите текст для анализа');
        return;
    }
    predictEmotion(text);
});

textInput.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        predictBtn.click();
    }
});

async function predictEmotion(text) {
    loadingDiv.classList.remove('hidden');
    resultDiv.classList.add('hidden');
    
    try {
        const response = await fetch('/predict_bert', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                text: text,
                model_type: parseInt(modelSelect.value)
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            showResult(data.emotion, data.confidence);
        } else {
            showError(data.error);
        }
    } catch (error) {
        showError('Ошибка соединения с сервером');
    } finally {
        loadingDiv.classList.add('hidden');
    }
}

function showResult(emotion, confidence) {
    resultContent.innerHTML = `
        <div class="result-card">
            <p><strong>Эмоция:</strong> ${emotion}</p>
            <p><strong>Уверенность:</strong> ${(confidence * 100).toFixed(2)}%</p>
        </div>
    `;
    resultDiv.classList.remove('hidden');
}

function showError(error) {
    resultContent.innerHTML = `<div class="error-card">Ошибка: ${error}</div>`;
    resultDiv.classList.remove('hidden');
}