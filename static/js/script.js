/**
 * Registration Card Scanner - Frontend Logic
 * Handles image upload, OCR scanning, and results display
 */

(function() {
    'use strict';

    // --- DOM References ---
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const scanBtn = document.getElementById('scanBtn');
    const resetBtn = document.getElementById('resetBtn');
    const spinner = document.getElementById('spinner');
    const progressSteps = document.getElementById('progressSteps');
    const placeholder = document.getElementById('placeholder');
    const resultsContainer = document.getElementById('resultsContainer');
    const fieldsGrid = document.getElementById('fieldsGrid');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const rawOcrToggle = document.getElementById('rawOcrToggle');
    const rawOcrContent = document.getElementById('rawOcrContent');
    const rawOcrSection = document.getElementById('rawOcrSection');
    const statusBadge = document.getElementById('statusBadge');
    const copyBtn = document.getElementById('copyBtn');
    const downloadBtn = document.getElementById('downloadBtn');
    const toastContainer = document.getElementById('toastContainer');

    // --- State ---
    let selectedFile = null;
    let currentResults = null;
    let isProcessing = false;

    // --- Initialization ---
    function init() {
        checkAPIStatus();
        bindEvents();
    }

    // --- API Health Check ---
    async function checkAPIStatus() {
        try {
            const res = await fetch('/health');
            if (res.ok) {
                setStatus('connected', 'API Connected');
            } else {
                setStatus('disconnected', 'API Error');
            }
        } catch {
            setStatus('disconnected', 'Server Offline');
        }
    }

    function setStatus(state, text) {
        statusBadge.className = 'status-badge ' + state;
        statusBadge.innerHTML = `<span class="status-dot"></span> ${text}`;
    }

    // --- Event Binding ---
    function bindEvents() {
        // File upload via click
        uploadZone.addEventListener('click', (e) => {
            if (uploadZone.classList.contains('has-image')) return;
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length) {
                handleFileSelect(e.target.files[0]);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            if (!uploadZone.classList.contains('has-image') && !isProcessing) {
                uploadZone.classList.add('dragover');
            }
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.classList.remove('dragover');
            if (uploadZone.classList.contains('has-image') || isProcessing) return;
            if (e.dataTransfer.files.length) {
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });

        // Scan button
        scanBtn.addEventListener('click', processImage);

        // Reset button
        resetBtn.addEventListener('click', resetAll);

        // Export buttons
        copyBtn.addEventListener('click', copyResults);
        downloadBtn.addEventListener('click', downloadResults);

        // Raw OCR toggle
        rawOcrToggle.addEventListener('click', toggleRawOcr);
    }

    // --- File Handling ---
    function handleFileSelect(file) {
        const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/tiff', 'image/bmp'];
        const maxSize = 50 * 1024 * 1024; // 50 MB

        if (!allowedTypes.includes(file.type) && !file.name.match(/\.(jpg|jpeg|png|webp|tiff|bmp)$/i)) {
            showToast('Please select a valid image file (JPEG, PNG, WebP, TIFF, BMP)', 'error');
            return;
        }

        if (file.size > maxSize) {
            showToast('File is too large. Maximum size is 50 MB.', 'error');
            return;
        }

        selectedFile = file;
        previewImage(file);
    }

    function previewImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.classList.add('visible');
            uploadZone.classList.add('has-image');
            uploadZone.querySelector('.upload-content').style.display = 'none';
            scanBtn.disabled = false;
            resetBtn.style.display = 'inline-flex';
            showToast('Image loaded successfully', 'success');
        };
        reader.readAsDataURL(file);
    }

    // --- Scanning ---
    async function processImage() {
        if (!selectedFile || isProcessing) return;

        isProcessing = true;
        scanBtn.disabled = true;
        placeholder.style.display = 'none';
        resultsContainer.classList.remove('visible');
        rawOcrSection.classList.remove('visible');
        spinner.classList.add('visible');
        progressSteps.classList.add('visible');

        // Start progress animation
        updateProgressStep(1);

        const formData = new FormData();
        formData.append('image', selectedFile);

        try {
            updateProgressStep(2);

            const response = await fetch('/api/extract', {
                method: 'POST',
                body: formData,
            });

            updateProgressStep(3);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Processing failed');
            }

            const data = await response.json();
            currentResults = data;

            updateProgressStep(4);

            // Small delay for UX polish
            await new Promise(r => setTimeout(r, 300));

            spinner.classList.remove('visible');
            progressSteps.classList.remove('visible');
            displayResults(data);

            if (data.success) {
                showToast('Scan completed successfully!', 'success');
            }

        } catch (error) {
            spinner.classList.remove('visible');
            progressSteps.classList.remove('visible');
            placeholder.style.display = 'flex';
            showToast(error.message || 'Failed to process image', 'error');
            console.error('Scan error:', error);
        } finally {
            isProcessing = false;
            scanBtn.disabled = false;
        }
    }

    function updateProgressStep(step) {
        const steps = progressSteps.querySelectorAll('span');
        steps.forEach((s, i) => {
            s.classList.remove('active', 'done');
            if (i + 1 < step) s.classList.add('done');
            else if (i + 1 === step) s.classList.add('active');
        });
    }

    // --- Results Display ---
    function displayResults(data) {
        resultsContainer.classList.add('visible');
        fieldsGrid.innerHTML = '';

        // Sort fields by field_number
        const fields = Object.entries(data.fields || {})
            .sort(([, a], [, b]) => a.field_number - b.field_number);

        fields.forEach(([key, field]) => {
            const card = document.createElement('div');
            card.className = `field-card ${field.value ? 'found' : 'not-found'}`;

            const confidencePercent = Math.round(field.confidence * 100);

            card.innerHTML = `
                <div class="field-number">${field.field_number}</div>
                <div class="field-content">
                    <div class="field-label">${field.label}</div>
                    <div class="field-value ${field.value ? '' : 'empty'}">
                        ${field.value || 'Not detected'}
                    </div>
                    ${field.value ? `<div class="field-confidence">Confidence: ${confidencePercent}%</div>` : ''}
                </div>
            `;

            fieldsGrid.appendChild(card);
        });

        // Update confidence badge
        const overallConf = data.confidence || 0;
        const confPercent = Math.round(overallConf * 100);
        let level = 'low';
        let label = `Confidence: ${confPercent}%`;
        if (confPercent >= 70) level = 'high';
        else if (confPercent >= 40) level = 'medium';

        confidenceBadge.className = `confidence-badge ${level}`;
        confidenceBadge.innerHTML = `<i class="fas fa-chart-line"></i> ${label}`;

        // Store raw OCR for toggle
        if (data.raw_ocr && data.raw_ocr.length) {
            rawOcrContent.textContent = JSON.stringify(data.raw_ocr, null, 2);
            rawOcrSection.classList.add('visible');
        }

        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // --- Raw OCR Toggle ---
    let rawOcrVisible = false;

    function toggleRawOcr() {
        rawOcrVisible = !rawOcrVisible;
        const rawContent = document.getElementById('rawOcrContent');
        rawContent.style.display = rawOcrVisible ? 'block' : 'none';
        rawOcrToggle.innerHTML = rawOcrVisible
            ? '<i class="fas fa-eye-slash"></i> Hide Raw OCR Output'
            : '<i class="fas fa-eye"></i> Show Raw OCR Output';
    }

    // --- Reset ---
    function resetAll() {
        selectedFile = null;
        currentResults = null;
        isProcessing = false;
        rawOcrVisible = false;

        imagePreview.src = '';
        imagePreview.classList.remove('visible');
        uploadZone.classList.remove('has-image');
        uploadZone.querySelector('.upload-content').style.display = '';
        uploadZone.classList.remove('dragover');

        scanBtn.disabled = true;
        resetBtn.style.display = 'none';
        placeholder.style.display = 'flex';
        resultsContainer.classList.remove('visible');
        rawOcrSection.classList.remove('visible');
        spinner.classList.remove('visible');
        progressSteps.classList.remove('visible');

        fileInput.value = '';
    }

    // --- Export Functions ---
    function copyResults() {
        if (!currentResults) return;

        const exportData = {};
        const fields = currentResults.fields || {};
        Object.entries(fields).forEach(([key, field]) => {
            exportData[key] = {
                label: field.label,
                value: field.value,
                confidence: field.confidence
            };
        });

        const text = JSON.stringify(exportData, null, 2);
        navigator.clipboard.writeText(text).then(() => {
            showToast('Results copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy to clipboard', 'error');
        });
    }

    function downloadResults() {
        if (!currentResults) return;

        const exportData = {
            scan_date: new Date().toISOString(),
            confidence: currentResults.confidence,
            fields: {}
        };

        const fields = currentResults.fields || {};
        Object.entries(fields).forEach(([key, field]) => {
            exportData.fields[key] = {
                label: field.label,
                value: field.value,
                confidence: field.confidence
            };
        });

        const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `registration-card-scan-${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        showToast('Results downloaded as JSON!', 'success');
    }

    // --- Toast Notifications ---
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = message;
        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(100%)';
            toast.style.transition = 'all 0.3s ease';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    // --- Bootstrap ---
    document.addEventListener('DOMContentLoaded', init);

})();
