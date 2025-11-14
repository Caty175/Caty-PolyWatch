const allowedExtensions = ['exe', 'dll', 'sys', 'js', 'vbs', 'ps1', 'py'];

const fileInput = document.getElementById('fileInput');
const dropZone = document.getElementById('dropZone');
const browseBtn = document.getElementById('browseBtn');
const scanBtn = document.getElementById('scanBtn');
const clearBtn = document.getElementById('clearBtn');
const uploadError = document.getElementById('uploadError');
const uploadSuccess = document.getElementById('uploadSuccess');
const fileSummary = document.getElementById('fileSummary');
const resultsList = document.getElementById('resultsList');
const resultsCount = document.getElementById('resultsCount');
const resultContent = document.getElementById('resultContent');
const resultStatusBadge = document.getElementById('resultStatusBadge');
const modelTrainingDate = document.getElementById('modelTrainingDate');

const clearMessages = () => {
    uploadError.textContent = '';
    uploadError.classList.remove('show');
    uploadSuccess.textContent = '';
    uploadSuccess.classList.remove('show');
};

const setError = (message) => {
    if (!message) return;
    uploadError.textContent = message;
    uploadError.classList.add('show');
};

const setSuccess = (message) => {
    if (!message) return;
    uploadSuccess.textContent = message;
    uploadSuccess.classList.add('show');
};

const formatBytes = (bytes) => {
    if (!bytes) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
};

const isAllowedExtension = (filename) => {
    if (!filename) return false;
    const ext = filename.split('.').pop()?.toLowerCase();
    return ext ? allowedExtensions.includes(ext) : false;
};

let selectedFile = null;

const resetSelection = () => {
    selectedFile = null;
    fileInput.value = '';
    fileSummary.innerHTML = '<span class="file-name">No file selected</span>';
    scanBtn.disabled = true;
    clearBtn.disabled = true;
};

const updateFileSummary = (file) => {
    if (!file) {
        resetSelection();
        return;
    }

    const extValid = isAllowedExtension(file.name);
    fileSummary.innerHTML = `
        <div class="file-summary-details">
            <span class="file-name">${file.name}</span>
            <span class="file-meta">${formatBytes(file.size)}</span>
        </div>
        <span class="file-chip ${extValid ? 'chip-valid' : 'chip-invalid'}">
            ${extValid ? 'Allowed type' : 'Unsupported type'}
        </span>
    `;

    scanBtn.disabled = !extValid;
    clearBtn.disabled = false;

    if (!extValid) {
        setError('This file type is not supported. Allowed extensions: .exe, .dll, .sys, .js, .vbs, .ps1, .py');
    }
};

const handleFiles = (files) => {
    clearMessages();
    if (!files || !files.length) return;
    const file = files[0];
    selectedFile = file;
    updateFileSummary(file);
};

if (browseBtn) {
    browseBtn.addEventListener('click', () => {
        fileInput?.click();
    });
}

if (fileInput) {
    fileInput.addEventListener('change', (event) => {
        const target = event.target;
        handleFiles(target.files);
    });
}

const activateDropZone = () => dropZone.classList.add('is-dragover');
const deactivateDropZone = () => dropZone.classList.remove('is-dragover');

if (dropZone) {
    dropZone.addEventListener('dragenter', (event) => {
        event.preventDefault();
        event.stopPropagation();
        activateDropZone();
    });

    dropZone.addEventListener('dragover', (event) => {
        event.preventDefault();
        event.stopPropagation();
        activateDropZone();
    });

    dropZone.addEventListener('dragleave', (event) => {
        event.preventDefault();
        event.stopPropagation();
        deactivateDropZone();
    });

    dropZone.addEventListener('drop', (event) => {
        event.preventDefault();
        event.stopPropagation();
        deactivateDropZone();
        const dt = event.dataTransfer;
        if (dt && dt.files) {
            handleFiles(dt.files);
        }
    });
}

if (clearBtn) {
    clearBtn.addEventListener('click', () => {
        clearMessages();
        resetSelection();
    });
}

const updateSidebarResults = (result) => {
    if (!resultsList || !resultsCount) return;
    const { filename, prediction, confidence, timestamp } = result;

    const listItem = document.createElement('li');
    listItem.className = `results-item ${prediction === 'malware' ? 'result-malware' : 'result-benign'}`;
    listItem.innerHTML = `
        <div class="result-item-header">
            <span class="result-item-name" title="${filename}">${filename}</span>
            <span class="result-item-prediction">${prediction.toUpperCase()}</span>
        </div>
        <div class="result-item-meta">
            <span>${(confidence * 100).toFixed(1)}% confidence</span>
            <span>${new Date(timestamp).toLocaleTimeString()}</span>
        </div>
    `;

    const placeholder = resultsList.querySelector('.results-placeholder');
    if (placeholder) {
        resultsList.innerHTML = '';
    }

    resultsList.prepend(listItem);

    const total = resultsList.querySelectorAll('li').length;
    resultsCount.textContent = String(total);
};

const renderResultContent = (result) => {
    if (!resultContent || !resultStatusBadge) return;
    const {
        filename,
        prediction,
        confidence,
        malware_probability,
        benign_probability,
        file_size,
        features_extracted,
        timestamp,
        top_features,
        random_forest_probability,
        lstm_probability,
        sandbox_analysis_performed,
    } = result;

    resultStatusBadge.textContent = prediction === 'malware' ? 'Malware' : 'Benign';
    resultStatusBadge.className = `result-badge ${prediction === 'malware' ? 'badge-malware' : 'badge-benign'}`;

    // Build model breakdown section - always show if random_forest_probability is available
    // Handle case where random_forest_probability might not be in response (fallback to malware_probability)
    const rfProb = random_forest_probability !== null && random_forest_probability !== undefined 
        ? random_forest_probability 
        : malware_probability; // Fallback to malware_probability if RF prob not available
    
    let modelBreakdownMarkup = '';
    
    if (rfProb !== null && rfProb !== undefined) {
        if (sandbox_analysis_performed && lstm_probability !== null && lstm_probability !== undefined) {
            // Both RF and LSTM results available (sandbox analysis performed)
            modelBreakdownMarkup = `
                <div class="model-breakdown">
                    <h3>Model Analysis</h3>
                    <div class="analysis-indicator">
                        <span class="indicator-icon">🔍</span>
                        <span class="indicator-text">Sandbox behavioral analysis performed</span>
                    </div>
                    <div class="model-probabilities">
                        <div class="model-prob-item">
                            <span class="model-label">Random Forest (30% weight)</span>
                            <span class="model-value">${(rfProb * 100).toFixed(1)}%</span>
                        </div>
                        <div class="model-prob-item">
                            <span class="model-label">LSTM Behavioral (70% weight)</span>
                            <span class="model-value">${(lstm_probability * 100).toFixed(1)}%</span>
                        </div>
                        <div class="model-prob-item model-prob-combined">
                            <span class="model-label"><strong>Combined Result</strong></span>
                            <span class="model-value"><strong>${(malware_probability * 100).toFixed(1)}%</strong></span>
                        </div>
                    </div>
                </div>
            `;
        } else {
            // Only RF result available (sandbox analysis not performed)
            // Display weighted result (RF * 30%) and show RF raw value separately
            const rfPercent = (rfProb * 100).toFixed(1);
            const weightedPercent = (malware_probability * 100).toFixed(1);
            modelBreakdownMarkup = `
                <div class="model-breakdown">
                    <h3>Model Analysis</h3>
                    <div class="model-probabilities">
                        <div class="model-prob-item">
                            <span class="model-label">Random Forest (30% weight)</span>
                            <span class="model-value">${rfPercent}%</span>
                        </div>
                        <div class="model-prob-item">
                            <span class="model-label">LSTM Behavioral (70% weight)</span>
                            <span class="model-value">Not triggered</span>
                        </div>
                        <div class="model-prob-item model-prob-combined">
                            <span class="model-label"><strong>Combined Result</strong></span>
                            <span class="model-value"><strong>${weightedPercent}%</strong></span>
                        </div>
                    </div>
                    ${rfProb < 0.60 ? `
                        <div class="analysis-note">
                            <span class="note-icon">ℹ️</span>
                            <span class="note-text">RF probability (${rfPercent}%) below 60% threshold. Sandbox analysis not triggered.</span>
                        </div>
                    ` : rfProb >= 0.60 ? `
                        <div class="analysis-note">
                            <span class="note-icon">⚠️</span>
                            <span class="note-text">RF probability (${rfPercent}%) meets threshold, but sandbox analysis was not available or failed.</span>
                        </div>
                    ` : ''}
                </div>
            `;
        }
    }

    const topFeaturesMarkup = Array.isArray(top_features) && top_features.length
        ? `
            <h3>Top Features</h3>
            <ul class="feature-list">
                ${top_features.map((item) => `
                    <li>
                        <span class="feature-name">${item.feature || item.name}</span>
                        <span class="feature-value">${item.value ?? 'n/a'}</span>
                        ${item.importance !== undefined ? `<span class="feature-importance">Importance: ${(item.importance * 100).toFixed(2)}%</span>` : ''}
                    </li>
                `).join('')}
            </ul>
        `
        : '';

    resultContent.innerHTML = `
        <div class="result-summary">
            <div>
                <h3>${filename}</h3>
                <p>Scanned at ${new Date(timestamp).toLocaleString()}</p>
            </div>
            <div class="result-metrics">
                <div>
                    <span class="metric-label">Confidence</span>
                    <span class="metric-value">${(confidence * 100).toFixed(1)}%</span>
                </div>
                <div>
                    <span class="metric-label">Malware Probability</span>
                    <span class="metric-value">${(malware_probability * 100).toFixed(1)}%</span>
                </div>
                <div>
                    <span class="metric-label">Benign Probability</span>
                    <span class="metric-value">${(benign_probability * 100).toFixed(1)}%</span>
                </div>
            </div>
        </div>
        <div class="result-details">
            <div>
                <h3>File Details</h3>
                <ul class="detail-list">
                    <li><span>File size</span><span>${formatBytes(file_size)}</span></li>
                    <li><span>Features extracted</span><span>${features_extracted}</span></li>
                    <li><span>File type</span><span>${filename.split('.').pop()?.toUpperCase()}</span></li>
                </ul>
                ${modelBreakdownMarkup}
            </div>
            <div>
                ${topFeaturesMarkup || '<p>No feature breakdown available for this scan.</p>'}
            </div>
        </div>
    `;
};

const fetchModelMetadata = async () => {
    if (!modelTrainingDate) return;
    try {
        const response = await fetch('/model/info');
        if (!response.ok) return;
        const data = await response.json();
        if (data.training_date) {
            modelTrainingDate.textContent = data.training_date;
        }
    } catch (error) {
        console.warn('Unable to fetch model info:', error);
    }
};

const submitScan = async () => {
    if (!selectedFile) return;
    clearMessages();

    if (!isAllowedExtension(selectedFile.name)) {
        setError('Unsupported file type. Please choose a PE file (.exe, .dll, .sys) or script (.js, .vbs, .ps1, .py).');
        return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    scanBtn.disabled = true;
    scanBtn.textContent = 'Scanning...';

    try {
        const response = await fetch('/scan/detailed', {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Scan failed. Please try again.');
        }

        // Debug: Log the result to see what fields are available
        console.log('Scan result:', result);
        console.log('RF probability:', result.random_forest_probability);
        console.log('LSTM probability:', result.lstm_probability);
        console.log('Sandbox performed:', result.sandbox_analysis_performed);

        setSuccess('Scan completed successfully.');
        renderResultContent(result);
        updateSidebarResults(result);
        resetSelection();
    } catch (error) {
        console.error('Scan error:', error);
        setError(error.message || 'An unexpected error occurred during scanning.');
    } finally {
        scanBtn.disabled = false;
        scanBtn.textContent = 'Scan File';
    }
};

if (scanBtn) {
    scanBtn.addEventListener('click', submitScan);
}

fetchModelMetadata();

