// ==========================================
// STATE MANAGEMENT
// ==========================================
let currentMode = 'simple';
let sessionId = null;
let spoilerProtectionEnabled = false;
let maxChapter = null;
let includeReference = false;

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function resetConversation() {
    if (!confirm("Start a new conversation? This will clear the current history and memory.")) {
        return;
    }

    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = '';

    if (currentMode === 'conversational') {
        sessionId = generateSessionId();
        const sessionDisplay = document.getElementById('sessionId');
        if (sessionDisplay) {
            sessionDisplay.textContent = sessionId.substr(0, 20) + '...';
        }
        addMessage('assistant', 'Started a new conversation session. I have forgotten our previous context.');
    } else {
        addMessage('assistant', 'Chat cleared. Ready for new questions.');
    }
}

// ==========================================
// MODE SWITCHING & UI CONTROL
// ==========================================

function setMode(mode) {
    currentMode = mode;

    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[onclick*="'${mode}'"]`).classList.add('active');

    if (mode === 'conversational') {
        if (!sessionId) sessionId = generateSessionId();
        document.getElementById('sessionInfo').style.display = 'block';
        document.getElementById('sessionId').textContent = sessionId.substr(0, 20) + '...';
        addMessage('assistant', `Conversational mode enabled. I'll remember our conversation context.`);
    } else {
        sessionId = null;
        document.getElementById('sessionInfo').style.display = 'none';
        addMessage('assistant', `Simple Q&A mode. Each question is independent.`);
    }
}

// ==========================================
// SPOILER PROTECTION CONTROLS
// ==========================================

function toggleSpoilerProtection() {
    const toggle = document.getElementById('spoilerToggle');
    const slider = document.getElementById('chapterSlider');
    const sliderContainer = document.getElementById('sliderContainer');
    const referenceContainer = document.getElementById('referenceContainer');
    const statusDisplay = document.getElementById('spoilerStatus');

    spoilerProtectionEnabled = toggle.checked;

    if (spoilerProtectionEnabled) {
        // ENABLE spoiler protection
        slider.disabled = false;
        sliderContainer.style.opacity = '1.0';
        sliderContainer.style.pointerEvents = 'auto';

        // Enable reference toggle
        referenceContainer.style.opacity = '1.0';
        referenceContainer.style.pointerEvents = 'auto';

        // Set initial chapter value
        let currentVal = parseInt(slider.value) || 10;
        let maxVal = parseInt(slider.max) || 50;

        if (currentVal <= 1) {
            currentVal = Math.min(10, maxVal);
            slider.value = currentVal;
        }

        maxChapter = currentVal;
        updateStatusDisplay();
    } else {
        // DISABLE spoiler protection
        slider.disabled = true;
        sliderContainer.style.opacity = '0.5';
        sliderContainer.style.pointerEvents = 'none';

        // Disable reference toggle
        referenceContainer.style.opacity = '0.5';
        referenceContainer.style.pointerEvents = 'none';

        maxChapter = null;
        statusDisplay.textContent = 'OFF - Full Book';
        statusDisplay.classList.remove('active');
    }
}

function updateChapterDisplay() {
    const slider = document.getElementById('chapterSlider');
    const display = document.getElementById('chapterValue');

    const val = parseInt(slider.value) || 1;
    maxChapter = val;
    display.textContent = maxChapter;

    updateStatusDisplay();
}

function updateStatusDisplay() {
    const statusDisplay = document.getElementById('spoilerStatus');
    const refCheckbox = document.getElementById('includeReference');
    includeReference = refCheckbox.checked;

    let status = `ON - Ch. 1-${maxChapter}`;
    if (includeReference) {
        status += ' + refs';
    }
    statusDisplay.textContent = status;
    statusDisplay.classList.add('active');
}

// ==========================================
// DOCUMENT MANAGEMENT
// ==========================================

async function uploadDocument() {
    const titleInput = document.getElementById('docTitle');
    const fileInput = document.getElementById('fileInput');
    const title = titleInput.value.trim();
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    if (title) formData.append('title', title);

    try {
        addMessage('assistant', `Uploading "${file.name}"... This may take a moment for large files.`);

        const response = await fetch('/api/v1/documents/upload-file', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const doc = await response.json();
            addMessage('assistant', `Document "${doc.title}" uploaded and processed successfully.`);
            titleInput.value = '';
            fileInput.value = '';
            loadDocuments();
            loadStatus();
        } else {
            const errorData = await response.json();
            addMessage('assistant', `Upload failed: ${errorData.detail}`);
        }
    } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
    }
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/v1/documents/list');
        const docs = await response.json();

        const listDiv = document.getElementById('documentList');
        const filterSelect = document.getElementById('documentFilter');

        if (docs.length === 0) {
            listDiv.innerHTML = '<div style="color: #666; font-size: 0.85em; text-align: center; padding: 20px;">No documents yet</div>';
            filterSelect.innerHTML = '<option value="" data-chapters="50">Search in: All Documents</option>';
            return;
        }

        // Sidebar List - show chapter count if available
        listDiv.innerHTML = docs.map(doc => {
            let chapterInfo = '';
            if (doc.total_chapters && doc.total_chapters > 0) {
                chapterInfo = ` (${doc.total_chapters} ch)`;
            }
            return `
                <div class="doc-item">
                    <div>
                        <div class="doc-title" title="${doc.title}">${doc.title}${chapterInfo}</div>
                        <div class="doc-filename" title="${doc.filename}">${doc.filename}</div>
                    </div>
                    <button class="delete-btn" onclick="deleteDocument(${doc.id})">Delete</button>
                </div>
            `;
        }).join('');

        // Dropdown - use total_chapters from API, fallback to 50
        filterSelect.innerHTML = '<option value="" data-chapters="50">Search in: All Documents</option>' +
            docs.map(doc => {
                // Use total_chapters if available and > 0, otherwise default to 50
                const chapters = (doc.total_chapters && doc.total_chapters > 0) ? doc.total_chapters : 50;
                return `<option value="${doc.id}" data-chapters="${chapters}">${doc.title}</option>`;
            }).join('');

        // Update slider when document selection changes
        filterSelect.onchange = function() {
            updateSliderMax(this);
        };

        // Initial update
        updateSliderMax(filterSelect);

    } catch (error) {
        console.error('Failed to load documents:', error);
    }
}

function updateSliderMax(selectElement) {
    const selectedOption = selectElement.options[selectElement.selectedIndex];
    const chapters = parseInt(selectedOption.getAttribute('data-chapters')) || 50;

    const slider = document.getElementById('chapterSlider');
    const oldMax = parseInt(slider.max);

    // Update max
    slider.max = chapters;

    // Clamp current value if it exceeds new max
    if (parseInt(slider.value) > chapters) {
        slider.value = chapters;
    }

    // If max changed significantly and spoiler is on, update display
    if (spoilerProtectionEnabled && oldMax !== chapters) {
        updateChapterDisplay();
        console.log(`Slider max updated: ${oldMax} → ${chapters}`);
    }
}

async function deleteDocument(id) {
    if (!confirm('Delete this document?')) return;
    try {
        const response = await fetch(`/api/v1/documents/${id}`, {method: 'DELETE'});
        if (response.ok) {
            loadDocuments();
            loadStatus();
            addMessage('assistant', 'Document deleted.');
        }
    } catch (error) {
        alert('Failed to delete document');
    }
}

// ==========================================
// CHAT INTERACTION
// ==========================================

async function askQuestion() {
    const input = document.getElementById('questionInput');
    const question = input.value.trim();

    if (!question) return;

    input.value = '';
    addMessage('user', question);

    const askBtn = document.getElementById('askBtn');
    askBtn.disabled = true;
    askBtn.innerHTML = '<span class="loading"></span>';

    try {
        const filterSelect = document.getElementById('documentFilter');
        const documentId = filterSelect.value;
        const refCheckbox = document.getElementById('includeReference');
        includeReference = refCheckbox.checked;

        // Build query params
        const params = new URLSearchParams();
        if (documentId) params.append('document_id', documentId);
        if (spoilerProtectionEnabled && maxChapter) {
            params.append('max_chapter', maxChapter);
            params.append('include_reference', includeReference);
        }

        let url, body;

        if (currentMode === 'conversational') {
            if (sessionId) params.set('session_id', sessionId);
            url = `/api/v1/conversation/ask?${params.toString()}`;
            body = {question: question};
        } else {
            url = `/api/v1/chat/ask?${params.toString()}`;
            body = {question: question, max_chunks: 3};
        }

        const response = await fetch(url, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(body)
        });

        const result = await response.json();

        if (result.error) {
            addMessage('assistant', `Error: ${result.error}`);
        } else {
            let answer = result.answer;

            // Add spoiler banner if active
            if (result.spoiler_filter_active) {
                let banner = `📖 *[Reading Ch. 1-${result.max_chapter}`;
                if (result.include_reference) {
                    banner += ' + reference material';
                }
                banner += ']*';
                answer = `${banner}\n\n${answer}`;
            }

            addMessage('assistant', answer, result.sources);
        }
    } catch (error) {
        addMessage('assistant', `Error: ${error.message}`);
    } finally {
        askBtn.disabled = false;
        askBtn.textContent = 'Send';
    }
}

function addMessage(role, content, sources = []) {
    const messagesDiv = document.getElementById('messages');
    const messageDiv = document.createElement('div');

    messageDiv.className = role === 'user' ? 'message user' : 'message assistant';

    // Format italic banner text
    const formattedContent = content.replace(/\*\[(.+?)\]\*/g, '<em style="color: #888; font-size: 0.85em;">[$1]</em>');
    let html = `<p>${formattedContent}</p>`;

    // Format sources
    if (sources && sources.length > 0) {
        const sourceMap = sources.reduce((acc, source) => {
            let key = source.chapter_title || source.document_title;
            if (source.is_reference) {
                key = `📚 ${key}`;
            }
            if (!acc[key]) acc[key] = [];
            if (!acc[key].includes(source.chunk_index)) acc[key].push(source.chunk_index);
            return acc;
        }, {});

        const sourceHtml = Object.keys(sourceMap).map(title => title).join(' • ');
        html += `<div class="compact-sources">${sourceHtml}</div>`;
    }

    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ==========================================
// STATUS & INITIALIZATION
// ==========================================

async function loadStatus() {
    try {
        const response = await fetch('/api/v1/chat/status');
        const status = await response.json();

        const statusBadge = document.getElementById('headerStatus');
        if (statusBadge) {
            if (status.status === 'ready') {
                statusBadge.textContent = '● System Ready';
                statusBadge.className = 'status-badge ready';
            } else {
                statusBadge.textContent = '○ Not Ready';
                statusBadge.className = 'status-badge not_ready';
            }
        }

        const docCount = document.getElementById('docCount');
        const chunkCount = document.getElementById('chunkCount');
        if (docCount) docCount.textContent = status.documents_loaded;
        if (chunkCount) chunkCount.textContent = status.total_chunks;

    } catch (error) {
        console.error('Failed to load status:', error);
        const statusBadge = document.getElementById('headerStatus');
        if(statusBadge) {
            statusBadge.textContent = '! Connection Lost';
            statusBadge.className = 'status-badge not_ready';
        }
    }
}

// Reference checkbox listener
document.addEventListener('DOMContentLoaded', function() {
    const refCheckbox = document.getElementById('includeReference');
    if (refCheckbox) {
        refCheckbox.addEventListener('change', function() {
            includeReference = this.checked;
            if (spoilerProtectionEnabled) {
                updateStatusDisplay();
            }
        });
    }
});

// Initial load
loadDocuments();
loadStatus();
setInterval(loadStatus, 30000);