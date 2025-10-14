let currentMode = 'simple';
let sessionId = null;

function generateSessionId() {
    return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

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


                    method: 'POST'
                });

                if (processResponse.ok) {
                    addMessage('assistant', `Document "${title}" uploaded and processed successfully.`);
                    titleInput.value = '';
                    fileInput.value = '';
                    loadDocuments();
                    loadStatus();
                } else {
                    alert('Document uploaded but processing failed');
                }
            } else {
                alert('Failed to upload document');
            }
        } catch (error) {
            alert('Error: ' + error.message);
        }
    };

    reader.readAsText(file);
}

async function loadDocuments() {
    try {
        const response = await fetch('/api/v1/documents/list');
        const docs = await response.json();

        const listDiv = document.getElementById('documentList');

        if (docs.length === 0) {
            listDiv.innerHTML = '<div style="color: #666; font-size: 0.85em; text-align: center; padding: 20px;">No documents yet</div>';
            return;
        }

        listDiv.innerHTML = docs.map(doc => `
            <div class="doc-item">
                <div>
                    <div class="doc-title">${doc.title}</div>
                    <div class="doc-filename">${doc.filename}</div>
                </div>
                <button class="delete-btn" onclick="deleteDocument(${doc.id})">Delete</button>
            </div>
        `).join('');
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
        let url, body;

        if (currentMode === 'conversational') {
            url = `/api/v1/conversation/ask?session_id=${sessionId}`;
            body = {question: question};
        } else {
            url = '/api/v1/chat/ask';
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
            addMessage('assistant', result.answer, result.sources);
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

    let html = `<p>${content}</p>`;

    if (sources && sources.length > 0) {
        html += '<div class="sources"><strong>Sources:</strong>';
        sources.forEach(source => {
            html += `<div class="source-item">→ ${source.document_title} (chunk ${source.chunk_index})</div>`;
        });
        html += '</div>';
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
                <div class="status-label">Documents</div>
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

// Initialize on page load
loadDocuments();
loadStatus();
setInterval(loadStatus, 30000);