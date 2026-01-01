/**
 * Human Interaction Modals
 * Handles all human review interactions with clickable selections
 */

// API_BASE is declared in pipeline.js
const humanModal = document.getElementById('humanModal');
const modalTitle = document.getElementById('modalTitle');
const modalContent = document.getElementById('modalContent');

let currentInputType = null;
let currentPipelineId = null;

function showHumanModal(data) {
    if (!currentPipelineId && typeof pipelineId !== 'undefined') {
        currentPipelineId = pipelineId;
    } else if (!currentPipelineId && window.pipelineId) {
        currentPipelineId = window.pipelineId;
    }


    currentInputType = data.input_type;

    switch (data.input_type) {
        case 'preprocessing_config':
            showPreprocessingModal(data.data);
            break;
        case 'model_selection':
            showModelSelectionModal(data.data);
            break;
        case 'evaluation_decision':
            showEvaluationModal(data.data);
            break;
        case 'inference_decision':
            showInferenceModal(data.data);
            break;
    }

    humanModal.classList.remove('hidden');
}

function hideHumanModal() {
    humanModal.classList.add('hidden');
}

// ============================================================================
// Preprocessing Config Modal
// ============================================================================

function showPreprocessingModal(data) {
    try {
        console.log('[DEBUG] showPreprocessingModal called with:', data);
        modalTitle.textContent = 'Review Preprocessing Configuration';

        // Robust default structure
        const safeConfig = data.config || {};
        const safeFeatures = safeConfig.features || {};
        safeFeatures.drop = safeFeatures.drop || [];
        safeConfig.features = safeFeatures;

        safeConfig.scaling = safeConfig.scaling || { method: 'standard' };
        safeConfig.dimensionality_reduction = safeConfig.dimensionality_reduction || { use_pca: false, variance_threshold: 0.95 };

        const config = safeConfig;
        const allColumns = [...(data.num_columns || []), ...(data.cat_columns || [])];

        modalContent.innerHTML = `
            <div class="space-y-6">
                <!-- Columns to Drop -->
                <div>
                    <h3 class="font-semibold mb-3">Select Columns to Drop</h3>
                    <div class="grid grid-cols-2 gap-2 max-h-60 overflow-y-auto bg-gray-900/50 p-4 rounded-lg">
                        ${allColumns.map(col => `
                            <label class="flex items-center space-x-2 cursor-pointer hover:bg-gray-700/50 p-2 rounded">
                                <input type="checkbox" value="${col}" 
                                       ${(config.features.drop && config.features.drop.includes(col)) ? 'checked' : ''}
                                       class="drop-column-checkbox">
                                <span class="text-sm">${col}</span>
                            </label>
                        `).join('')}
                    </div>
                </div>
                
                <!-- Scaling Method -->
                <div>
                    <h3 class="font-semibold mb-3">Scaling Method</h3>
                    <div class="flex gap-3">
                        <label class="flex-1 cursor-pointer">
                            <input type="radio" name="scaler" value="standard" 
                                   ${config.scaling.method === 'standard' ? 'checked' : ''}
                                   class="mr-2">
                            <span>Standard</span>
                        </label>
                        <label class="flex-1 cursor-pointer">
                            <input type="radio" name="scaler" value="minmax" 
                                   ${config.scaling.method === 'minmax' ? 'checked' : ''}
                                   class="mr-2">
                            <span>MinMax</span>
                        </label>
                        <label class="flex-1 cursor-pointer">
                            <input type="radio" name="scaler" value="robust" 
                                   ${config.scaling.method === 'robust' ? 'checked' : ''}
                                   class="mr-2">
                            <span>Robust</span>
                        </label>
                    </div>
                </div>
                
                <!-- PCA -->
                <div>
                    <h3 class="font-semibold mb-3">Dimensionality Reduction (PCA)</h3>
                    <label class="flex items-center space-x-2 cursor-pointer mb-3">
                        <input type="checkbox" id="usePCA" 
                               ${config.dimensionality_reduction.use_pca ? 'checked' : ''}>
                        <span>Use PCA</span>
                    </label>
                    <div id="pcaOptions" class="${config.dimensionality_reduction.use_pca ? '' : 'hidden'}">
                        <label class="block text-sm mb-2">Variance Threshold</label>
                        <input type="number" id="pcaVariance" 
                               value="${config.dimensionality_reduction.variance_threshold}" 
                               min="0.5" max="1" step="0.05"
                               class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg">
                    </div>
                </div>
                
                <!-- Buttons -->
                <div class="flex gap-3 pt-4">
                    <button onclick="submitPreprocessingConfig()" 
                            class="flex-1 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold">
                        Accept Configuration
                    </button>
                </div>
            </div>
        `;

        // Toggle PCA options
        const pcaCheckbox = document.getElementById('usePCA');
        if (pcaCheckbox) {
            pcaCheckbox.addEventListener('change', (e) => {
                document.getElementById('pcaOptions').classList.toggle('hidden', !e.target.checked);
            });
        }

        // Store original config
        window.currentConfig = config;

    } catch (e) {
        console.error('Error in showPreprocessingModal:', e);
        alert('Error displaying preprocessing options: ' + e.message);
    }
}

async function submitPreprocessingConfig() {
    const dropColumns = Array.from(document.querySelectorAll('.drop-column-checkbox:checked'))
        .map(cb => cb.value);

    const scaler = document.querySelector('input[name="scaler"]:checked').value;
    const usePCA = document.getElementById('usePCA').checked;
    const pcaVariance = parseFloat(document.getElementById('pcaVariance').value);

    // Get original config and update
    const config = JSON.parse(JSON.stringify(window.currentConfig || {}));
    config.features.drop = dropColumns;
    config.scaling.method = scaler;
    config.dimensionality_reduction.use_pca = usePCA;
    config.dimensionality_reduction.variance_threshold = pcaVariance;

    await sendHumanResponse({
        accepted: true,
        config: config
    });

    hideHumanModal();
}

// ============================================================================
// Model Selection Modal
// ============================================================================

function showModelSelectionModal(data) {
    modalTitle.textContent = 'Select Models to Train';

    modalContent.innerHTML = `
        <div class="space-y-6">
            <div class="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
                <p class="text-sm"><strong>Task:</strong> ${data.task_type}</p>
                <p class="text-sm"><strong>Metric:</strong> ${data.primary_metric}</p>
            </div>
            
            <div>
                <h3 class="font-semibold mb-3">Select Models (uncheck to drop)</h3>
                <div class="space-y-3" id="modelsContainer">
                    ${data.models.map((model, idx) => `
                        <div class="bg-gray-700/50 rounded-lg overflow-hidden">
                            <label class="flex items-start space-x-3 p-4 cursor-pointer hover:bg-gray-700">
                                <input type="checkbox" value="${idx}" checked class="model-checkbox mt-1">
                                <div class="flex-1">
                                    <div class="font-medium">${model.name}</div>
                                    <div class="text-sm text-gray-400">${model.rationale}</div>
                                    <div class="text-xs text-gray-500 mt-1">
                                        ${model.params.map(p => `${p.name}=${p.value}`).join(', ')}
                                    </div>
                                    <button type="button" 
                                            onclick="toggleHyperparameters(${idx})" 
                                            class="text-xs text-blue-400 hover:text-blue-300 mt-2">
                                        ⚙️ Edit Hyperparameters
                                    </button>
                                </div>
                            </label>
                            <div id="params-${idx}" class="hidden border-t border-gray-600 p-4 bg-gray-800/50">
                                <h4 class="text-sm font-semibold mb-3">Hyperparameters</h4>
                                <div class="space-y-2">
                                    ${model.params.map((p, pidx) => `
                                        <div class="flex items-center gap-3">
                                            <label class="text-sm w-32">${p.name}:</label>
                                            <input type="${typeof p.value === 'number' ? 'number' : 'text'}" 
                                                   value="${p.value}" 
                                                   data-model="${idx}" 
                                                   data-param="${pidx}"
                                                   class="param-input flex-1 px-3 py-1 bg-gray-700 border border-gray-600 rounded text-sm"
                                                   ${typeof p.value === 'number' ? 'step="any"' : ''}>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            
            <div class="flex gap-3 pt-4">
                <button onclick="submitModelSelection()" 
                        class="flex-1 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold">
                    Train Selected Models
                </button>
            </div>
        </div>
    `;

    // Store original data
    window.currentModelsData = data;
}

// Toggle hyperparameter editing section
window.toggleHyperparameters = function (modelIdx) {
    const paramsDiv = document.getElementById(`params-${modelIdx}`);
    paramsDiv.classList.toggle('hidden');
};

async function submitModelSelection() {
    const selectedIndices = Array.from(document.querySelectorAll('.model-checkbox:checked'))
        .map(cb => parseInt(cb.value));

    // Clone models data
    const selectedModels = window.currentModelsData.models
        .filter((_, idx) => selectedIndices.includes(idx))
        .map(m => JSON.parse(JSON.stringify(m))); // Deep clone

    // Update parameters from edited inputs
    document.querySelectorAll('.param-input').forEach(input => {
        const modelIdx = parseInt(input.dataset.model);
        const paramIdx = parseInt(input.dataset.param);

        // Only update if this model is selected
        const selectedModelArrayIdx = selectedIndices.indexOf(modelIdx);
        if (selectedModelArrayIdx !== -1) {
            const originalValue = window.currentModelsData.models[modelIdx].params[paramIdx].value;
            let newValue = input.value;

            // Convert to number if original was a number
            if (typeof originalValue === 'number') {
                newValue = parseFloat(newValue);
            }

            selectedModels[selectedModelArrayIdx].params[paramIdx].value = newValue;
        }
    });

    await sendHumanResponse({
        accepted: true,
        models: selectedModels,
        primary_metric: window.currentModelsData.primary_metric
    });

    hideHumanModal();
}

// ============================================================================
// Evaluation Decision Modal
// ============================================================================

function showEvaluationModal(data) {
    modalTitle.textContent = 'Model Evaluation Results';

    const results = data.evaluation_results;
    const agent = data.agent_response;

    modalContent.innerHTML = `
        <div class="space-y-6">
            <div class="bg-gray-700/50 rounded-lg p-4">
                <h3 class="font-semibold mb-3">Model Ranking</h3>
                ${results.model_ranking.map(m => `
                    <div class="flex justify-between py-2 border-b border-gray-600">
                        <span>${m.rank}. ${m.model_name}</span>
                        <span class="font-mono">${m.score.toFixed(4)}</span>
                    </div>
                `).join('')}
            </div>
            
            <div class="bg-blue-900/30 border border-blue-700 rounded-lg p-4">
                <p class="text-sm mb-2"><strong>Best Model:</strong> ${results.best_model}</p>
                <p class="text-sm mb-2"><strong>Score:</strong> ${results.best_score.toFixed(4)}</p>
                <p class="text-sm"><strong>Confidence:</strong> ${results.confidence}</p>
            </div>
            
            ${results.warnings.length > 0 ? `
                <div class="bg-yellow-900/30 border border-yellow-700 rounded-lg p-4">
                    <p class="font-semibold mb-2">⚠️ Warnings</p>
                    ${results.warnings.map(w => `<p class="text-sm">• ${w}</p>`).join('')}
                </div>
            ` : ''}
            
            <div class="bg-gray-700/50 rounded-lg p-4">
                <p class="font-semibold mb-2">🤖 Agent Recommendation: ${agent.decision}</p>
                <p class="text-sm text-gray-300">${agent.reasoning}</p>
            </div>
            
            <div class="flex gap-3 pt-4">
                <button onclick="submitEvaluationDecision('accept')" 
                        class="flex-1 py-3 bg-green-600 hover:bg-green-700 rounded-lg font-semibold">
                    Accept Models
                </button>
                <button onclick="submitEvaluationDecision('retrain')" 
                        class="flex-1 py-3 bg-orange-600 hover:bg-orange-700 rounded-lg font-semibold">
                    Retrain
                </button>
            </div>
        </div>
    `;
}

async function submitEvaluationDecision(decision) {
    await sendHumanResponse({ decision });
    hideHumanModal();
}

// ============================================================================
// Inference Decision Modal
// ============================================================================

function showInferenceModal(data) {
    modalTitle.textContent = 'Run Inference?';

    modalContent.innerHTML = `
        <div class="space-y-6">
            <div>
                <h3 class="font-semibold mb-3">Select Models for Inference</h3>
                <div class="space-y-2">
                    ${data.trained_models.map((model, idx) => `
                        <label class="flex items-center space-x-3 bg-gray-700/50 p-3 rounded-lg cursor-pointer hover:bg-gray-700">
                            <input type="checkbox" value="${idx}" class="inference-model-checkbox">
                            <span>${model.name}</span>
                        </label>
                    `).join('')}
                </div>
            </div>
            
            <div>
                <h3 class="font-semibold mb-3">Test Dataset</h3>
                <input type="file" id="testDatasetInput" accept=".csv" 
                       class="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-lg">
            </div>
            
            <div class="flex gap-3 pt-4">
                <button onclick="submitInferenceDecision(true)" 
                        class="flex-1 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold">
                    Run Inference
                </button>
                <button onclick="submitInferenceDecision(false)" 
                        class="flex-1 py-3 bg-gray-600 hover:bg-gray-700 rounded-lg font-semibold">
                    Skip Inference
                </button>
            </div>
        </div>
    `;

    window.currentInferenceData = data;
}

async function submitInferenceDecision(runInference) {
    if (!runInference) {
        await sendHumanResponse({ run_inference: false });
        hideHumanModal();
        return;
    }

    const selectedIndices = Array.from(document.querySelectorAll('.inference-model-checkbox:checked'))
        .map(cb => parseInt(cb.value));

    if (selectedIndices.length === 0) {
        alert('Please select at least one model');
        return;
    }

    const selectedModels = window.currentInferenceData.trained_models
        .filter((_, idx) => selectedIndices.includes(idx))
        .map(m => m.name);

    const testFile = document.getElementById('testDatasetInput').files[0];
    if (!testFile) {
        alert('Please select a test dataset');
        return;
    }

    // Upload test file
    const formData = new FormData();
    formData.append('file', testFile);

    const uploadResponse = await fetch(`${API_BASE}/files/upload`, {
        method: 'POST',
        body: formData
    });

    const uploadData = await uploadResponse.json();

    await sendHumanResponse({
        run_inference: true,
        selected_models: selectedModels,
        test_dataset_path: `uploads/${uploadData.filename}`
    });

    hideHumanModal();
}

// ============================================================================
// Helper Functions
// ============================================================================

async function sendHumanResponse(responseData) {
    try {
        console.log('[DEBUG] Sending response to:', `${API_BASE}/pipeline/respond/${currentPipelineId}`);
        console.log('[DEBUG] Response data:', responseData);

        const response = await fetch(`${API_BASE}/pipeline/respond/${currentPipelineId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ response_data: responseData })
        });

        console.log('[DEBUG] Response status:', response.status);
        const responseText = await response.text();
        console.log('[DEBUG] Response body:', responseText);

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}: ${responseText}`);
        }

        addLog('✓ Response sent, resuming pipeline...', 'text-green-400');
    } catch (error) {
        console.error('Error sending response:', error);
        alert('Failed to send response: ' + error.message);
    }
}
