/**
 * Frontend JavaScript for AgenticML Upload Page
 */

const API_BASE = '/api';

let selectedFile = null;

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const removeFileBtn = document.getElementById('removeFile');
const targetColumn = document.getElementById('targetColumn');
const startPipelineBtn = document.getElementById('startPipeline');
const loadingState = document.getElementById('loadingState');
const autoWarning = document.getElementById('autoWarning');

// File Upload Handlers
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('border-blue-500', 'bg-gray-700/30');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('border-blue-500', 'bg-gray-700/30');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('border-blue-500', 'bg-gray-700/30');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFileSelect(e.target.files[0]);
    }
});

removeFileBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    selectedFile = null;
    fileInput.value = '';
    fileInfo.classList.add('hidden');
    dropZone.classList.remove('hidden');
    validateForm();
});

function handleFileSelect(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please select a CSV file');
        return;
    }

    if (file.size > 100 * 1024 * 1024) {
        alert('File size must be less than 100MB');
        return;
    }

    selectedFile = file;
    fileName.textContent = `${file.name} (${formatFileSize(file.size)})`;
    dropZone.classList.add('hidden');
    fileInfo.classList.remove('hidden');
    validateForm();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

// Task Type Selection
document.querySelectorAll('input[name="taskType"]').forEach(radio => {
    radio.addEventListener('change', (e) => {
        if (e.target.value === 'auto') {
            autoWarning.classList.remove('hidden');
        } else {
            autoWarning.classList.add('hidden');
        }
        validateForm();
    });
});

// Target Column Input
targetColumn.addEventListener('input', validateForm);

// Form Validation
function validateForm() {
    const isValid = selectedFile && targetColumn.value.trim() !== '';
    startPipelineBtn.disabled = !isValid;
}

// Start Pipeline
startPipelineBtn.addEventListener('click', async () => {
    if (!selectedFile || !targetColumn.value.trim()) {
        return;
    }

    const taskType = document.querySelector('input[name="taskType"]:checked').value;

    try {
        // Show loading state
        startPipelineBtn.disabled = true;
        loadingState.classList.remove('hidden');

        // Upload file
        const formData = new FormData();
        formData.append('file', selectedFile);


        console.log('Uploading file:', selectedFile.name);
        const uploadResponse = await fetch(`${API_BASE}/files/upload`, {
            method: 'POST',
            body: formData
        });

        console.log('Upload response status:', uploadResponse.status);
        if (!uploadResponse.ok) {
            const errorText = await uploadResponse.text();
            console.error('Upload error:', errorText);
            throw new Error('File upload failed: ' + errorText);
        }

        const uploadData = await uploadResponse.json();
        console.log('Upload successful:', uploadData);

        // Start pipeline
        console.log('Starting pipeline with:', {
            dataset_filename: uploadData.filename,
            target_column: targetColumn.value.trim(),
            task_type: taskType
        });
        const pipelineResponse = await fetch(`${API_BASE}/pipeline/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                dataset_filename: uploadData.filename,
                target_column: targetColumn.value.trim(),
                task_type: taskType
            })
        });

        if (!pipelineResponse.ok) {
            throw new Error('Failed to start pipeline');
        }

        const pipelineData = await pipelineResponse.json();

        // Redirect to pipeline view
        window.location.href = `pipeline.html?id=${pipelineData.pipeline_id}`;

    } catch (error) {
        console.error('Error:', error);
        console.error('Error details:', {
            name: error.name,
            message: error.message,
            stack: error.stack
        });
        alert('Failed to start pipeline: ' + error.message + '\n\nCheck browser console for details.');
        startPipelineBtn.disabled = false;
        loadingState.classList.add('hidden');
    }
});
