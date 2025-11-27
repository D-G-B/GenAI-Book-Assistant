let currentMode = 'simple';
let sessionId = null;
let spoilerProtectionEnabled = false;
let maxChapter = null;

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`.mode-btn[onclick*="'${mode}'"]`).classList.add('active');

    if (mode === 'conversational') {
        sessionId = generateSessionId();
        document.getElementById('sessionInfo').style.display = 'block';
        document.getElementById('sessionId').textContent = sessionId.substr(0, 20) + '...';
        addMessage('assistant', `Conversational mode enabled. I'll remember our conversation context.`);
    } else {
        sessionId = null;
        document.getElementById('sessionInfo').style.display = 'none';
        addMessage('assistant', `Simple Q&A mode. Each question is independent.`);
    }
}

function toggleSpoilerProtection() {
    const toggle = document.getElementById('spoilerToggle');
    const sliderContainer = document.getElementById('sliderContainer');
    const statusDisplay = document.getElementById('spoilerStatus');

    spoilerProtectionEnabled = toggle.checked;

    if (spoilerProtectionEnabled) {
        sliderContainer.style.display = 'block';
        maxChapter = parseInt(document.getElementById('chapterSlider').value);
        statusDisplay.textContent = `ON - Up to Ch. ${maxChapter}`;
        statusDisplay.classList.add('active');
    } else {
        sliderContainer.style.display = 'none';
        maxChapter = null;
        statusDisplay.textContent = 'OFF - Full Book';
        statusDisplay.classList.remove('active');
    }
}

function updateChapterDisplay() {
    const slider = document.getElementById('chapterSlider');
    const display = document.getElementById('chapterValue');
    const statusDisplay = document.getElementById('spoilerStatus');

    maxChapter = parseInt(slider.value);
    display.textContent = maxChapter;
    statusDisplay.textContent = `ON - Up to Ch. ${maxChapter}`;
}

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
    if (title) {
        formData.append('title', title);
    }

    try {
        const response = await fetch('/api/v1/documents/upload-file', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const doc = await response.json();

            const processResponse = await fetch(`/api/v1/documents/${doc.id}/process`, {
                method: 'POST'
            });

            if (processResponse.ok) {
                addMessage('assistant', `Document "${doc.title}" uploaded and processed successfully.`);
                titleInput.value = '';
                fileInput.value = '';
                loadDocuments();
                loadStatus();
            } else {
                const errorData = await processResponse.json();
                addMessage('assistant', `Document uploaded but processing failed: ${errorData.detail || 'Unknown error'}`);
            }
        } else {
            const errorData = await response.json();
            alert(`Failed to upload document: ${errorData.detail || 'Unknown error'}`);
        }
    } catch (error) {
        alert('Error: ' + error.message);
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
            filterSelect.innerHTML = '<option value="">Search in: All Documents</option>';
            return;
        }

        listDiv.innerHTML = docs.map(doc => {
            const chapterInfo = doc.total_chapters ? ` (${doc.total_chapters} ch.)` : '';
            return `
                <div class="doc-item">
                    <div>
                        <div class="doc-title">${doc.title}${chapterInfo}</div>
                        <div class="doc-filename">${doc.filename}</div>
                    </div>
                    <button class="delete-btn" onclick="deleteDocument(${doc.id})">Delete</button>
                </div>
            `;
        }).join('');

        filterSelect.innerHTML = '<option value="">Search in: All Documents</option>' +
            docs.map(doc => `<option value="${doc.id}">${doc.title}</option>`).join('');

        // Update slider max based on document chapters
        const maxChapters = Math.max(...docs.map(d => d.total_chapters || 50));
        if (maxChapters > 0) {
            document.getElementById('chapterSlider').max = maxChapters;
        }

    } catch (error) {
        console.error('Failed to load documents:', error);
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

        let url, body;

        // Build query params
        const params = new URLSearchParams();
        if (documentId) {
            params.append('document_id', documentId);
        }
        if (spoilerProtectionEnabled && maxChapter) {
            params.append('max_chapter', maxChapter);
        }
        const queryString = params.toString() ? `?${params.toString()}` : '';

        if (currentMode === 'conversational') {
            if (sessionId) {
                params.set('session_id', sessionId);
            }
            url = `/api/v1/conversation/ask?${params.toString()}`;
            body = {question: question};
        } else {
            url = `/api/v1/chat/ask${queryString}`;
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
            // Add spoiler filter indicator to response if active
            let answer = result.answer;
            if (result.spoiler_filter_active) {
                answer = `📖 *[Searching chapters 1-${result.max_chapter} + reference material]*\n\n${answer}`;
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

    // Convert markdown-style italics for spoiler indicator
    const formattedContent = content.replace(/\*\[(.+?)\]\*/g, '<em style="color: #888; font-size: 0.85em;">[$1]</em>');

    let html = `<p>${formattedContent}</p>`;

    if (sources && sources.length > 0) {
        const sourceMap = sources.reduce((acc, source) => {
            const key = source.chapter_title || source.document_title;
            if (!acc[key]) {
                acc[key] = [];
            }
            if (!acc[key].includes(source.chunk_index)) {
                acc[key].push(source.chunk_index);
            }
            return acc;
        }, {});

        const sourceHtml = Object.keys(sourceMap).map(title => {
            const chunks = sourceMap[title].sort((a, b) => a - b).join(', ');
            return `${title} (${chunks})`;
        }).join(' • ');

        html += `<div class="compact-sources">${sourceHtml}</div>`;
    }

    messageDiv.innerHTML = html;
    messagesDiv.appendChild(messageDiv);
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

async function loadStatus() {
    try {
        const response = await fetch('/api/v1/chat/status');
        const status = await response.json();

        document.getElementById('status').innerHTML = `
            <div class="status-item">
                <div class="status-value">${status.documents_loaded}</div>
                <div class="status-label">Docs</div>
            </div>
            <div class="status-item">
                <div class="status-value">${status.total_chunks}</div>
                <div class="status-label">Chunks</div>
            </div>
            <div class="status-item">
                <div class="status-value">${status.status}</div>
                <div class="status-label">Status</div>
            </div>
        `;
    } catch (error) {
        console.error('Failed to load status:', error);
    }
}

// Initialize
loadDocuments();
loadStatus();
setInterval(loadStatus, 30000);