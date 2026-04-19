let currentModel = null;
let currentTransform = null;
let currentSlice = 0;
let maxSlice = 5;
let basePath = null;

// Загрузка списка моделей при старте
async function loadModels() {
    const response = await fetch('/api/embolism/images');
    const data = await response.json();
    
    if (!data.success) {
        console.error('Failed to load models:', data.error);
        return;
    }
    
    basePath = data.base_path;
    const modelSelect = document.getElementById('modelSelect');
    
    modelSelect.innerHTML = '<option value="">-- Выберите модель --</option>';
    
    data.models.forEach(model => {
        const option = document.createElement('option');
        option.value = model.name;
        option.textContent = model.name.replace(/_/g, ' ').toUpperCase();
        option.dataset.transforms = JSON.stringify(model.transforms);
        modelSelect.appendChild(option);
    });
    
    modelSelect.disabled = false;
    document.getElementById('loading').style.display = 'none';
}

// При выборе модели
document.getElementById('modelSelect').addEventListener('change', async (e) => {
    currentModel = e.target.value;
    if (!currentModel) return;
    
    const option = e.target.selectedOptions[0];
    const transforms = JSON.parse(option.dataset.transforms);
    
    // Заполняем трансформации
    const transformSelect = document.getElementById('transformSelect');
    transformSelect.innerHTML = '';
    transforms.forEach(t => {
        const opt = document.createElement('option');
        opt.value = t.name;
        opt.textContent = t.name.toUpperCase();
        transformSelect.appendChild(opt);
    });
    
    document.getElementById('transformSelector').style.display = 'block';
    
    // Автоматически выбираем первую трансформацию
    if (transforms.length > 0) {
        transformSelect.value = transforms[0].name;
        currentTransform = transforms[0].name;
        maxSlice = transforms[0].slices.length - 1;
        document.getElementById('sliceSlider').max = maxSlice;
        document.getElementById('sliceSelector').style.display = 'block';
        updateSliceDisplay();
        updateImages();
    }
});

// При смене трансформации
document.getElementById('transformSelect').addEventListener('change', (e) => {
    currentTransform = e.target.value;
    updateImages();
});

// При смене среза
document.getElementById('sliceSlider').addEventListener('input', (e) => {
    currentSlice = parseInt(e.target.value);
    updateSliceDisplay();
    updateImages();
});

function updateSliceDisplay() {
    document.getElementById('sliceValue').textContent = `${currentSlice + 1}/${maxSlice + 1}`;
}

async function updateImages() {
    if (!currentModel || !currentTransform) return;
    
    const ctUrl = `${basePath}/PAT034/${currentModel}/${currentTransform}/slice_${currentSlice}_ct.png`;
    const gtUrl = `${basePath}/PAT034/${currentModel}/${currentTransform}/slice_${currentSlice}_gt.png`;
    const predUrl = `${basePath}/PAT034/${currentModel}/${currentTransform}/slice_${currentSlice}_pred.png`;
    
    document.getElementById('ctImage').src = ctUrl;
    document.getElementById('gtImage').src = gtUrl;
    document.getElementById('predImage').src = predUrl;
    
    document.getElementById('imagesContainer').style.display = 'block';
}

// Запуск
loadModels();