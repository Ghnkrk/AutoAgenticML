/**
 * Pipeline View Page Logic
 */

const API_BASE = '/api';
window.API_BASE = API_BASE;

// Get pipeline ID from URL
const urlParams = new URLSearchParams(window.location.search);
const pipelineId = urlParams.get('id');
window.pipelineId = pipelineId;

// DOM Elements
const pipelineIdEl = document.getElementById('pipelineId');
const statusText = document.getElementById('statusText');
const currentNode = document.getElementById('currentNode');
const statusIcon = document.getElementById('statusIcon');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const logsContainer = document.getElementById('logsContainer');
const logs = document.getElementById('logs');
const toggleLogsBtn = document.getElementById('toggleLogs');
const downloadsSection = document.getElementById('downloadsSection');
const summarySection = document.getElementById('summarySection');
const summaryContent = document.getElementById('summaryContent');

// State
let logsVisible = true;
let wsClient;

// Initialize
if (!pipelineId) {
    alert('No pipeline ID provided');
    window.location.href = 'index.html';
} else {
    pipelineIdEl.textContent = `Pipeline ID: ${pipelineId}`;
    initializeWebSocket();
}

function initializeWebSocket() {
    wsClient = new PipelineWebSocket(pipelineId, handleWebSocketMessage);
    wsClient.connect();

    // Fetch initial status in case pipeline is already complete
    fetch(`${API_BASE}/pipeline/status/${pipelineId}`)
        .then(response => response.json())
        .then(data => {
            if (data) {
                updateStatus(data);

                // If there's a pending human request, show it!
                if (data.pending_human_request) {
                    console.log("Restoring pending human input request:", data.pending_human_request);
                    showHumanModal(data.pending_human_request);
                }
            }
        })
        .catch(err => console.error("Failed to fetch initial status:", err));
}

function handleWebSocketMessage(data) {
    console.log('Received:', data);

    switch (data.type) {
        case 'status':
            updateStatus(data);
            break;
        case 'node_start':
            handleNodeStart(data);
            break;
        case 'node_complete':
            handleNodeComplete(data);
            break;
        case 'log':
            addLog(data.message);
            break;
        case 'human_input_request':
            showHumanModal(data);
            break;
        case 'error':
            handleError(data);
            break;
    }
}

// Status update handler
// Status update handler
function updateStatus(data) {
    statusText.textContent = data.message || data.status;

    if (data.status === 'completed') {
        const spinner = document.getElementById('statusSpinner');
        if (spinner) {
            // Stop spinning and change to checkmark
            const container = document.getElementById('statusIconContainer');
            if (container) {
                container.innerHTML = '<svg class="h-8 w-8 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>';
            }
        }

        // Show downloads and summary
        // Always load downloads from API now (rollback)
        loadDownloads();

        const resultData = data.result || data;
        const summary = resultData.pipeline_summary || data.summary;
        if (summary) {
            showSummary(summary);
        }

    } else if (data.status === 'failed') {
        const container = document.getElementById('statusIconContainer');
        if (container) {
            container.innerHTML = '<svg class="h-8 w-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
        }
    }
}

function handleNodeStart(data) {
    currentNode.textContent = `Current: ${data.node}`;
    addLog(`▶ Starting ${data.node}...`, 'text-blue-400');
}

function handleNodeComplete(data) {
    addLog(`✓ Completed ${data.node}`, 'text-green-400');
}

function addLog(message, className = 'text-gray-300') {
    const logEntry = document.createElement('div');
    logEntry.className = `${className} mb-1`;
    logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    logs.appendChild(logEntry);

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function handleError(data) {
    statusText.textContent = 'Pipeline Failed';
    const container = document.getElementById('statusIconContainer');
    if (container) {
        container.innerHTML = '<svg class="h-8 w-8 text-red-500" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>';
    }

    addLog(`✗ Error: ${data.message}`, 'text-red-400');
}

// Toggle logs visibility
toggleLogsBtn.addEventListener('click', () => {
    logsVisible = !logsVisible;
    logsContainer.classList.toggle('hidden');
    toggleLogsBtn.textContent = logsVisible ? 'Hide Details' : 'Show Details';
});

// ... handleNodeStart, handleNodeComplete, addLog, handleError ... (kept same)

// Load downloads (Rolled back to API fetch)
async function loadDownloads() {
    try {
        // Load models
        const modelsResponse = await fetch(`${API_BASE}/files/list/models`);
        const models = await modelsResponse.json();

        let modelsHtml = '';
        if (models.length > 0) {
            modelsHtml = models.map(m => `
                <div class="bg-gray-700/50 rounded-lg p-4">
                    <div class="flex items-center justify-between mb-2">
                        <span class="font-medium">${m.model}</span>
                        <button onclick="downloadFile('model', '${m.model}')" class="text-blue-400 hover:text-blue-300">
                            <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                        </button>
                    </div>
                    ${m.metadata ? `<button onclick="downloadFile('metadata', '${m.metadata}')" class="text-xs text-gray-400 hover:text-gray-300">Download Metadata</button>` : ''}
                </div>
            `).join('');
        } else {
            modelsHtml = '<p class="text-gray-400 text-sm">No models trained yet.</p>';
        }

        document.getElementById('modelsDownload').innerHTML = `
            <h4 class="font-medium mb-3">Trained Models</h4>
            ${modelsHtml}
        `;

        // Load predictions
        const predsResponse = await fetch(`${API_BASE}/files/list/predictions`);
        const predictions = await predsResponse.json();

        let predsHtml = '';
        if (predictions.length > 0) {
            predsHtml = predictions.map(p => `
                <div class="bg-gray-700/50 rounded-lg p-4 flex items-center justify-between">
                    <span class="font-medium">${p.filename}</span>
                    <button onclick="downloadFile('prediction', '${p.filename}')" class="text-blue-400 hover:text-blue-300">
                        <svg class="h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                        </svg>
                    </button>
                </div>
            `).join('');
        } else {
            predsHtml = '<p class="text-gray-400 text-sm">No predictions available.</p>';
        }

        document.getElementById('predictionsDownload').innerHTML = `
            <h4 class="font-medium mb-3">Predictions</h4>
            ${predsHtml}
        `;

        downloadsSection.classList.remove('hidden');
    } catch (error) {
        console.error('Failed to load downloads:', error);
    }
}

function downloadFile(type, filename) {
    window.location.href = `${API_BASE}/files/download/${type}/${filename}`;
}

async function showSummary(summary) {
    summarySection.classList.remove('hidden');
    summaryContent.innerHTML = '';

    // Enhanced markdown to HTML conversion
    const convertMarkdown = (text) => {
        // Table support
        const lines = text.split('\n');
        let inTable = false;
        let htmlLines = [];

        for (let i = 0; i < lines.length; i++) {
            let line = lines[i].trim();

            if (line.startsWith('|')) {
                if (!inTable) {
                    inTable = true;
                    htmlLines.push('<div class="overflow-x-auto my-4"><table class="table-auto w-full text-left border-collapse border border-gray-600">');
                    // Header row
                    const cells = line.split('|').filter(c => c.trim());
                    htmlLines.push('<thead><tr class="bg-gray-700/50">');
                    htmlLines.push(cells.map(c => `<th class="border border-gray-600 px-4 py-2 font-semibold text-blue-300">${c.trim()}</th>`).join(''));
                    htmlLines.push('</tr></thead><tbody>');
                    continue;
                } else if (line.includes('---')) {
                    continue; // Skip separator line
                } else {
                    // Body row
                    const cells = line.split('|').filter(c => c.trim());
                    htmlLines.push(`<tr class="hover:bg-gray-700/30">${cells.map(c => `<td class="border border-gray-600 px-4 py-2 text-gray-300">${c.trim()}</td>`).join('')}</tr>`);
                    continue;
                }
            } else if (inTable) {
                inTable = false;
                htmlLines.push('</tbody></table></div>');
            }

            // Normal text processing
            let processed = line
                .replace(/^### (.*$)/, '<h3 class="text-xl font-bold mt-6 mb-3 text-blue-400">$1</h3>')
                .replace(/^## (.*$)/, '<h2 class="text-2xl font-bold mt-8 mb-4 text-purple-400">$1</h2>')
                .replace(/^# (.*$)/, '<h1 class="text-3xl font-bold mt-10 mb-5 text-blue-500">$1</h1>')
                .replace(/^(🚀|📊|🛠️|🏆|🔄|🔮|💡|🧠)(.*$)/, '<div class="text-2xl font-bold mt-6 mb-3 text-gradient">$1$2</div>')
                .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-white">$1</strong>')
                .replace(/\*(.*?)\*/g, '<em class="italic text-gray-300">$1</em>');

            if (processed === line && line.length > 0) {
                htmlLines.push(`<p class="mb-2">${processed}</p>`);
            } else {
                htmlLines.push(processed);
            }
        }
        if (inTable) htmlLines.push('</tbody></table></div>');

        return htmlLines.join('');
    };

    // Render HTML directly (streaming tables is complex/error-prone)
    const html = convertMarkdown(summary);
    summaryContent.innerHTML = html;

    // Auto-scroll to bottom
    summaryContent.scrollTop = summaryContent.scrollHeight;
}
