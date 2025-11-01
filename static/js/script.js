// Replace the getBasePath function at the beginning of the file
function getBasePath() {
    // Get the origin (protocol + domain + port)
    const origin = window.location.origin; // e.g., "http://localhost:8502"
    
    // Get any application base path (in case the app is deployed in a subdirectory)
    let appPath = '';
    const pathnameParts = window.location.pathname.split('/');
    
    // Remove the last part if it's a file or empty
    if (pathnameParts.length > 0) {
        // Check if last part contains a dot (likely a file) or is empty
        if (pathnameParts[pathnameParts.length - 1].includes('.') || 
            pathnameParts[pathnameParts.length - 1] === '') {
            pathnameParts.pop();
        }
        appPath = pathnameParts.join('/');
    }
    
    console.log("API Base URL:", origin + appPath);
    return origin + appPath;
}

// Move updateFilesList outside DOMContentLoaded to make it globally accessible
async function updateFilesList() {
    try {
        const response = await fetch(getBasePath() + '/files', {
            credentials: 'same-origin'
        });
        
        // Check for 401 Unauthorized
        if (response.status === 401) {
            // Redirect to login page
            window.location.href = getBasePath() + '/login';
            return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        const filesDiv = document.getElementById('loadedFiles');
        
        if (Object.keys(data.files).length === 0) {
            filesDiv.innerHTML = `
                <div class="no-files">
                    <span class="material-symbols-outlined">folder_off</span>
                    <p>No documents uploaded yet</p>
                </div>`;
            return;
        }
        
        filesDiv.innerHTML = Object.entries(data.files)
            .map(([filename, chunks]) => `
                <div class="file-item" data-filename="${filename}">
                    <button class="file-action" onclick="removeFile('${filename}')">
                        <span class="material-symbols-outlined">close</span>
                    </button>
                    <div class="file-icon">
                        <span class="material-symbols-outlined">description</span>
                    </div>
                    <div class="file-details">
                        <div class="file-name" title="${filename}">
                            ${filename}
                        </div>
                        <div class="file-chunks">
                            <span class="material-symbols-outlined">data_object</span>
                            ${chunks} chunks
                        </div>
                    </div>
                </div>
            `).join('');
    } catch (error) {
        console.error('Error updating files list:', error);
    }
}

// Add this outside the DOMContentLoaded event
async function removeFile(filename) {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
        return;
    }

    try {
        const response = await fetch(getBasePath() + '/delete-file', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ filename }),
            credentials: 'same-origin'
        });

        if (!response.ok) {
            if (response.status === 401) {
                // Redirect to login page if not authorized
                window.location.href = getBasePath() + '/login';
                return;
            }
            throw new Error('Server responded with an error');
        }

        const data = await response.json();

        if (data.error) {
            throw new Error(data.error);
        }

        // Find and remove the specific file item from DOM
        const fileItem = document.querySelector(`.file-item[data-filename="${filename}"]`);
        if (fileItem) {
            fileItem.remove();
        }

        // Update files list
        await updateFilesList();

        // Show success message
        const toast = document.createElement('div');
        toast.className = 'toast-message success';
        toast.textContent = 'File deleted successfully';
        document.body.appendChild(toast);

        // Remove toast after 3 seconds
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 3000);

    } catch (error) {
        console.error('Error deleting file:', error);
        alert(error.message || 'Error deleting file');
    }
}

// Chat history management
let currentChatId = null;
let chatMessages = [];
const MAX_CHAT_HISTORY = 10;

// Generate chat ID
function generateChatId() {
    return 'chat_' + Date.now();
}

// Load chat history from localStorage
function loadChatHistory() {
    try {
        const history = localStorage.getItem('chatHistory');
        return history ? JSON.parse(history) : [];
    } catch (error) {
        console.error('Error loading chat history:', error);
        return [];
    }
}

// Save chat history to localStorage
function saveChatHistory(history) {
    try {
        localStorage.setItem('chatHistory', JSON.stringify(history));
    } catch (error) {
        console.error('Error saving chat history:', error);
    }
}

// Generate a title for the chat based on the first user message
function generateChatTitle(userMessage) {
    if (!userMessage) return "New Chat";
    
    // Take the first few words (max 5) from the first user message
    const words = userMessage.split(' ');
    let title = words.slice(0, 5).join(' ');
    
    // Add ellipsis if truncated
    if (words.length > 5) {
        title += '...';
    }
    
    return title;
}

// Save current chat to history
function saveCurrentChat() {
    if (chatMessages.length === 0) return; // Don't save empty chats
    
    const history = loadChatHistory();
    
    // Find user's first message for the title
    const firstUserMessage = chatMessages.find(msg => msg.role === 'user');
    const title = firstUserMessage ? generateChatTitle(firstUserMessage.content) : "New Chat";
    
    const chatEntry = {
        id: currentChatId,
        title: title,
        timestamp: new Date().toISOString(),
        messages: chatMessages
    };
    
    // Check if we're updating an existing chat
    const existingIndex = history.findIndex(item => item.id === currentChatId);
    if (existingIndex !== -1) {
        // Update existing chat
        history[existingIndex] = chatEntry;
    } else {
        // Add new chat to the beginning of the array
        history.unshift(chatEntry);
        
        // Limit to MAX_CHAT_HISTORY entries
        if (history.length > MAX_CHAT_HISTORY) {
            history.pop(); // Remove the oldest chat
        }
    }
    
    saveChatHistory(history);
    updateHistorySidebar();
}

// Start a new chat with server-side memory clearing
async function startNewChat() {
    try {
        // Call the server to start a new chat session
        const response = await fetch(getBasePath() + '/new-chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            credentials: 'same-origin'
        });
        
        if (response.status === 401) {
            window.location.href = getBasePath() + '/login';
            return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('New chat session started:', data.session_id);
        
        // Save current chat if it has messages (for local history)
        if (chatMessages.length > 0) {
            saveCurrentChat();
        }
        
        // Clear current chat UI and local state
        chatMessages = [];
        currentChatId = generateChatId();
        
        // Clear chat messages UI
        const chatMessagesEl = document.getElementById('chatMessages');
        chatMessagesEl.innerHTML = '';
        
        // Update history sidebar
        updateHistorySidebar();
        
        // Show confirmation message
        const toast = document.createElement('div');
        toast.className = 'toast-message success';
        toast.textContent = 'New chat started with fresh memory';
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.classList.add('fade-out');
            setTimeout(() => toast.remove(), 300);
        }, 2000);
        
    } catch (error) {
        console.error('Error starting new chat:', error);
        // Fallback to local-only new chat
        if (chatMessages.length > 0) {
            saveCurrentChat();
        }
        chatMessages = [];
        currentChatId = generateChatId();
        const chatMessagesEl = document.getElementById('chatMessages');
        chatMessagesEl.innerHTML = '';
        updateHistorySidebar();
    }
}

// Move formatResponse up here, before displayMessage
function formatResponse(text) {
    // Split text into lines
    let lines = text.split('\n');
    let formattedContent = '';
    let inList = false;

    for (let line of lines) {
        line = line.trim();
        if (!line) continue;

        // Handle headings
        if (line.match(/^[A-Z\s]{4,}:/)) {
            if (inList) {
                formattedContent += '</ul>';
                inList = false;
            }
            const headingText = line.replace(':', '');
            formattedContent += `<h2 class="response-heading">${headingText}</h2>`;
        }
        // Handle subheadings
        else if (line.match(/^[A-Z][a-z\s]+:/)) {
            if (inList) {
                formattedContent += '</ul>';
                inList = false;
            }
            const subheadingText = line.replace(':', '');
            formattedContent += `<h3 class="response-subheading">${subheadingText}</h3>`;
        }
        // Handle bullet points
        else if (line.startsWith('•') || line.startsWith('-')) {
            if (!inList) {
                formattedContent += '<ul class="response-list">';
                inList = true;
            }
            const bulletText = line.substring(1).trim();
            formattedContent += `<li>${bulletText}</li>`;
        }
        // Regular paragraphs
        else {
            if (inList) {
                formattedContent += '</ul>';
                inList = false;
            }
            formattedContent += `<p class="response-paragraph">${line}</p>`;
        }
    }

    if (inList) {
        formattedContent += '</ul>';
    }

    return formattedContent;
}

// Create a separate display function just for rendering messages without affecting history
function displayMessage(content, type) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    // Check if content contains PPT markers
    if (type === 'assistant' && content.includes('[PPT_CONTENT_START]')) {
        // Extract PPT content
        const pptContent = content.replace('[PPT_CONTENT_START]', '').replace('[PPT_CONTENT_END]', '');
        
        // Format the content
        const formattedContent = formatResponse(pptContent);
        bubble.innerHTML = formattedContent;

        // Add PPT download button
        const pptButton = document.createElement('button');
        pptButton.className = 'ppt-button';
        pptButton.innerHTML = `
            <span class="material-symbols-outlined">slideshow</span>
            Download Presentation
        `;
        pptButton.onclick = () => generatePPT(pptContent);
        
        messageDiv.appendChild(bubble);
        messageDiv.appendChild(pptButton);
    } else if (type === 'assistant') {
        bubble.innerHTML = formatResponse(content);
        messageDiv.appendChild(bubble);
    } else {
        bubble.textContent = content;
        messageDiv.appendChild(bubble);
    }

    chatMessages.appendChild(messageDiv);
    
    // Auto-scroll to ensure visibility
    setTimeout(() => {
        chatMessages.scrollTop = chatMessages.scrollHeight;
        // Additional scroll to ensure visibility
        const lastMessage = chatMessages.lastElementChild;
        if (lastMessage) {
            lastMessage.scrollIntoView({
                behavior: 'smooth',
                block: 'end',
                inline: 'nearest'
            });
        }
    }, 100);

    return messageDiv;
}

// Add generatePPT function globally, outside of DOMContentLoaded event
async function generatePPT(content) {
    try {
        const response = await fetch(getBasePath() + '/generate-ppt', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ content }),
            credentials: 'same-origin'
        });
        
        // Check for 401 Unauthorized
        if (response.status === 401) {
            // Redirect to login page
            window.location.href = getBasePath() + '/login';
            return;
        }
        
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        
        const data = await response.json();
        if (data.error) {
            throw new Error(data.error);
        }

        // Create temporary link to download PPT
        const link = document.createElement('a');
        link.href = data.download_url;
        link.download = 'presentation.pptx';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
    } catch (error) {
        console.error('Error generating PPT:', error);
        alert('Error generating presentation');
    }
}

// Delete a chat from history
function deleteChat(chatId, event) {
    // Stop the click event from bubbling up to the parent elements
    event.stopPropagation();
    
    // Ask for confirmation
    if (!confirm("Are you sure you want to delete this chat history?")) {
        return;
    }
    
    const history = loadChatHistory();
    
    // Find and remove the chat with the given ID
    const updatedHistory = history.filter(item => item.id !== chatId);
    
    // If we're deleting the current chat, clear the current chat and start a new one
    if (chatId === currentChatId) {
        chatMessages = [];
        currentChatId = generateChatId();
        
        // Clear the UI
        const chatMessagesEl = document.getElementById('chatMessages');
        chatMessagesEl.innerHTML = '';
    }
    
    // Save the updated history
    saveChatHistory(updatedHistory);
    
    // Update the sidebar
    updateHistorySidebar();
}

// Load a specific chat from history
function loadChat(chatId) {
    const history = loadChatHistory();
    const chat = history.find(item => item.id === chatId);
    
    if (!chat) return false;
    
    // Save current chat before switching
    if (chatMessages.length > 0 && currentChatId !== chatId) {
        saveCurrentChat();
    }
    
    // Set as current chat
    currentChatId = chatId;
    chatMessages = [...chat.messages];
    
    // Clear and rebuild UI
    const chatMessagesEl = document.getElementById('chatMessages');
    chatMessagesEl.innerHTML = '';
    
    // Add messages to UI
    chatMessages.forEach(message => {
        if (message.role === 'user') {
            displayMessage(message.content, 'user');
        } else if (message.role === 'assistant') {
            displayMessage(message.content, 'assistant');
        }
    });
    
    // Update history sidebar to highlight the current chat
    updateHistorySidebar();
    
    return true;
}

// Update history sidebar with current chats
function updateHistorySidebar() {
    const historyList = document.getElementById('historyList');
    const historyEmpty = document.getElementById('historyEmpty');
    const history = loadChatHistory();
    
    historyList.innerHTML = '';
    
    if (history.length === 0) {
        historyEmpty.style.display = 'flex';
        return;
    }
    
    historyEmpty.style.display = 'none';
    
    history.forEach(chat => {
        const date = new Date(chat.timestamp);
        const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        
        const historyItem = document.createElement('div');
        historyItem.className = 'history-item';
        if (chat.id === currentChatId) {
            historyItem.classList.add('active');
        }
        
        historyItem.innerHTML = `
            <div class="history-item-content">
                <div class="history-item-title">${chat.title}</div>
                <div class="history-item-date">${formattedDate}</div>
            </div>
            <button class="history-delete-btn" title="Delete chat">
                <span class="material-symbols-outlined">delete</span>
            </button>
        `;
        
        // Add click handler to load chat
        historyItem.addEventListener('click', () => {
            loadChat(chat.id);
            // Close sidebar on mobile
            if (window.innerWidth < 768) {
                document.getElementById('historySidebar').classList.remove('open');
            }
        });
        
        // Add click handler to delete button
        const deleteBtn = historyItem.querySelector('.history-delete-btn');
        deleteBtn.addEventListener('click', (e) => deleteChat(chat.id, e));
        
        historyList.appendChild(historyItem);
    });
}

document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const messageInput = document.getElementById('messageInput');
    const sendButton = document.getElementById('sendButton');
    const chatButton = document.getElementById('chatButton');
    const filesButton = document.getElementById('filesButton');
    const fileInput = document.getElementById('fileInput');
    const dropzone = document.getElementById('dropzone');
    const chatSection = document.getElementById('chatSection');
    const documentsSection = document.getElementById('documentsSection');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const contentTypeSelector = document.getElementById('contentTypeSelector');
    const modeSelector = document.getElementById('modeSelector');

    // Auto-resize textarea function
    function autoResizeTextarea() {
        messageInput.style.height = 'auto';
        const newHeight = Math.min(messageInput.scrollHeight, 120); // 120px max height
        messageInput.style.height = newHeight + 'px';
    }

    // Initialize textarea height
    autoResizeTextarea();

    // Add event listeners for textarea resizing
    messageInput.addEventListener('input', autoResizeTextarea);

    // Add a paste event handler to handle large pastes
    messageInput.addEventListener('paste', function() {
        // Use setTimeout to let the paste complete first
        setTimeout(autoResizeTextarea, 0);
    });

    // Navigation
    chatButton.addEventListener('click', () => {
        chatSection.classList.add('active');
        documentsSection.classList.remove('active');
        mcqSection.classList.remove('active'); // Ensure MCQ section is hidden
        
        chatButton.classList.add('active');
        filesButton.classList.remove('active');
        mcqButton.classList.remove('active');
        
        document.getElementById('mainHeader').style.display = 'block'; // Show header
        document.querySelector('.input-area').style.display = 'flex'; // Show input area
    });

    filesButton.addEventListener('click', () => {
        documentsSection.classList.add('active');
        chatSection.classList.remove('active');
        mcqSection.classList.remove('active'); // Ensure MCQ section is hidden
        
        filesButton.classList.add('active');
        chatButton.classList.remove('active');
        mcqButton.classList.remove('active');
        
        document.getElementById('mainHeader').style.display = 'none'; // Hide header
        document.querySelector('.input-area').style.display = 'none'; // Hide input area
    });

    // MCQ tab navigation with improved isolation
    const mcqButton = document.getElementById('mcqButton');
    const mcqSection = document.getElementById('mcqSection');
    
    if (mcqButton && mcqSection) {
        mcqButton.addEventListener('click', () => {
            chatSection.classList.remove('active');
            documentsSection.classList.remove('active');
            mcqSection.classList.add('active');
            
            chatButton.classList.remove('active');
            filesButton.classList.remove('active');
            mcqButton.classList.add('active');
            
            document.getElementById('mainHeader').style.display = 'none'; // Hide header
            document.querySelector('.input-area').style.display = 'none'; // Hide input area
        });
    }
    
    // File Upload Handling
    function handleFiles(files) {
        Array.from(files).forEach(file => {
            if (file.type === 'application/pdf') {
                uploadFile(file);
            } else {
                alert('Please upload PDF files only');
            }
        });
    }

    function formatFileSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
    }

    function createProgressBar(filename, filesize) {
        const container = document.createElement('div');
        container.className = 'upload-progress-container';
        container.innerHTML = `
            <div class="upload-progress-header">
                <span class="upload-progress-title">Uploading File</span>
                <span class="upload-progress-status">Processing...</span>
            </div>
            <div class="upload-progress-bar-container">
                <div class="upload-progress-bar" style="width: 0%"></div>
            </div>
            <div class="upload-progress-details">
                <div class="upload-progress-info">
                    <span class="upload-progress-filename">${filename}</span>
                    <span class="upload-progress-size">0 MB / ${formatFileSize(filesize)}</span>
                </div>
                <span class="upload-progress-percentage">0%</span>
            </div>
        `;
        document.body.appendChild(container);
        
        // Make visible after a small delay for animation
        setTimeout(() => container.classList.add('visible'), 100);
        
        return container;
    }

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        const progressContainer = createProgressBar(file.name, file.size);
        const progressBar = progressContainer.querySelector('.upload-progress-bar');
        const progressStatus = progressContainer.querySelector('.upload-progress-status');
        const progressPercentage = progressContainer.querySelector('.upload-progress-percentage');
        const progressSize = progressContainer.querySelector('.upload-progress-size');

        try {
            const xhr = new XMLHttpRequest();
            
            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable) {
                    const progress = Math.round((e.loaded * 100) / e.total);
                    const loadedSize = formatFileSize(e.loaded);
                    const totalSize = formatFileSize(e.total);
                    
                    progressBar.style.width = `${progress}%`;
                    progressPercentage.textContent = `${progress}%`;
                    progressSize.textContent = `${loadedSize} / ${totalSize}`;
                    
                    if (progress === 100) {
                        progressStatus.textContent = 'Processing file...';
                    }
                }
            });

            xhr.onload = function() {
                try {
                    if (xhr.status === 401) {
                        // Redirect to login page if not authorized
                        window.location.href = getBasePath() + '/login';
                        return;
                    }
                    
                    if (xhr.status >= 200 && xhr.status < 300) {
                        const response = JSON.parse(xhr.responseText);
                        
                        // Show success state with chunks info
                        progressBar.style.width = '100%';
                        progressStatus.textContent = 'Complete!';
                        progressPercentage.textContent = '100%';
                        
                        if (response.chunks) {
                            // Add chunks info to progress container
                            const infoText = progressContainer.querySelector('.upload-progress-info');
                            infoText.innerHTML += `<span class="upload-progress-chunks">${response.chunks} chunks created</span>`;
                        }

                        updateFilesList();
                        
                        // Remove progress bar after success
                        setTimeout(() => {
                            progressContainer.classList.remove('visible');
                            setTimeout(() => progressContainer.remove(), 300);
                            
                            // Show success toast with number of chunks
                            const toast = document.createElement('div');
                            toast.className = 'toast-message success';
                            toast.textContent = `File processed successfully with ${response.chunks || 'multiple'} chunks`;
                            document.body.appendChild(toast);
                            
                            // Remove toast after 4 seconds
                            setTimeout(() => {
                                toast.classList.add('fade-out');
                                setTimeout(() => toast.remove(), 300);
                            }, 4000);
                        }, 2000);
                    } else {
                        throw new Error('Server responded with an error');
                    }
                } catch (err) {
                    // Show error state
                    progressStatus.textContent = 'Failed';
                    progressContainer.style.backgroundColor = '#fee2e2';
                    progressBar.style.backgroundColor = '#ef4444';
                    
                    console.error('Error uploading file:', err);
                    
                    // Add error details to progress container
                    const infoText = progressContainer.querySelector('.upload-progress-info');
                    infoText.innerHTML += `<span class="upload-progress-error">${err.message || 'Error uploading file'}</span>`;
                    
                    // Remove progress bar after error (longer display time)
                    setTimeout(() => {
                        progressContainer.classList.remove('visible');
                        setTimeout(() => progressContainer.remove(), 300);
                    }, 5000);
                }
            };

            xhr.onerror = function() {
                // Show error state
                progressStatus.textContent = 'Failed';
                progressContainer.style.backgroundColor = '#fee2e2';
                progressBar.style.backgroundColor = '#ef4444';
                
                console.error('Network error during upload');
                
                // Add error details to progress container
                const infoText = progressContainer.querySelector('.upload-progress-info');
                infoText.innerHTML += `<span class="upload-progress-error">Network error during upload</span>`;
                
                // Remove progress bar after error
                setTimeout(() => {
                    progressContainer.classList.remove('visible');
                    setTimeout(() => progressContainer.remove(), 300);
                }, 5000);
            };

            xhr.open('POST', getBasePath() + '/upload');
            xhr.withCredentials = true; // Include credentials
            xhr.timeout = 180000; // 3 minutes timeout for large files
            xhr.send(formData);
        } catch (error) {
            // Show error state
            progressStatus.textContent = 'Failed';
            progressContainer.style.backgroundColor = '#fee2e2';
            progressBar.style.backgroundColor = '#ef4444';
            
            console.error('Error uploading file:', error);
            
            // Add error details to progress container
            const infoText = progressContainer.querySelector('.upload-progress-info');
            infoText.innerHTML += `<span class="upload-progress-error">${error.message || 'Error uploading file'}</span>`;
            
            // Remove progress bar after error
            setTimeout(() => {
                progressContainer.classList.remove('visible');
                setTimeout(() => progressContainer.remove(), 300);
            }, 5000);
        }
    }

    // Chat Handling with memory awareness
    async function sendMessage() {
        const message = messageInput.value.trim();
        if (!message) return;

        // Get content parameters
        const contentParams = {
            content_type: contentTypeSelector.value,
            mode: modeSelector.value
        };

        // Add user message
        addMessage(message, 'user');
        messageInput.value = '';
        
        // Reset textarea height after clearing
        messageInput.style.height = 'auto';

        // Add loading message
        const loadingMsgId = addLoadingMessage();

        try {
            const response = await fetch(getBasePath() + '/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: message,
                    content_params: contentParams
                }),
                credentials: 'same-origin'
            });
            
            // Check for 401 Unauthorized
            if (response.status === 401) {
                // Redirect to login page
                window.location.href = getBasePath() + '/login';
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Remove loading message
            removeLoadingMessage(loadingMsgId);
            
            if (data.error) {
                throw new Error(data.error);
            }

            // Log session ID for debugging
            if (data.session_id) {
                console.log('Current session ID:', data.session_id);
            }

            addMessage(data.response, 'assistant', data.metadata);
        } catch (error) {
            // Remove loading message if error occurs
            removeLoadingMessage(loadingMsgId);
            addMessage('Error: ' + error.message, 'error');
            console.error('API Error:', error);
        }
    }

    // Add content suggestion functionality
    async function suggestContent() {
        const topicInput = document.getElementById('topicInput').value.trim();
        const contentTypeForSuggestion = document.getElementById('suggestionType').value;
        
        if (!topicInput) {
            alert('Please enter a topic');
            return;
        }
        
        try {
            showLoading('Generating content ideas...');
            
            const response = await fetch(getBasePath() + '/suggest-content', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    topic: topicInput,
                    content_type: contentTypeForSuggestion
                }),
                credentials: 'same-origin'
            });
            
            // Check for 401 Unauthorized
            if (response.status === 401) {
                hideLoading();
                // Redirect to login page
                window.location.href = getBasePath() + '/login';
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            hideLoading();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Display suggestions in the suggestion area
            const suggestionsArea = document.getElementById('suggestionsArea');
            suggestionsArea.innerHTML = `<h3>Content Ideas for "${topicInput}"</h3>`;
            suggestionsArea.innerHTML += `<div class="suggestions-content">${formatResponse(data.suggestions)}</div>`;
            
        } catch (error) {
            hideLoading();
            alert('Error generating suggestions: ' + error.message);
            console.error('API Error:', error);
        }
    }

    function addLoadingMessage() {
        const chatMessages = document.getElementById('chatMessages');
        const loadingDiv = document.createElement('div');
        const uniqueId = 'loading-' + Date.now();
        loadingDiv.id = uniqueId;
        loadingDiv.className = 'message assistant loading';
        loadingDiv.innerHTML = `
            <div class="message-bubble loading-bubble">
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
                Just a moment...
            </div>
        `;
        chatMessages.appendChild(loadingDiv);
        
        // Auto-scroll for loading message with a small delay
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }, 50);
        
        return uniqueId;
    }

    function removeLoadingMessage(loadingId) {
        const loadingDiv = document.getElementById(loadingId);
        if (loadingDiv) {
            loadingDiv.remove();
        }
    }

    // UI Functions
    function showLoading(message) {
        document.getElementById('loadingText').textContent = message;
        loadingOverlay.style.display = 'flex';
    }

    function hideLoading() {
        loadingOverlay.style.display = 'none';
    }

    async function addMessage(content, type) {
        const chatMessages = document.getElementById('chatMessages');
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';

        // Check if content contains PPT markers
        if (type === 'assistant' && content.includes('[PPT_CONTENT_START]')) {
            // Extract PPT content
            const pptContent = content.replace('[PPT_CONTENT_START]', '').replace('[PPT_CONTENT_END]', '');
            
            // Format the content
            const formattedContent = formatResponse(pptContent);
            bubble.innerHTML = formattedContent;

            // Add PPT download button
            const pptButton = document.createElement('button');
            pptButton.className = 'ppt-button';
            pptButton.innerHTML = `
                <span class="material-symbols-outlined">slideshow</span>
                Download Presentation
            `;
            pptButton.onclick = () => generatePPT(pptContent);
            
            messageDiv.appendChild(bubble);
            messageDiv.appendChild(pptButton);
        } else if (type === 'assistant') {
            bubble.innerHTML = formatResponse(content);
            messageDiv.appendChild(bubble);
        } else {
            bubble.textContent = content;
            messageDiv.appendChild(bubble);
        }

        chatMessages.appendChild(messageDiv);
        
        // Enhanced auto-scroll with a small delay to ensure complete rendering
        setTimeout(() => {
            chatMessages.scrollTop = chatMessages.scrollHeight;
            
            // Additional scroll to ensure visibility
            const lastMessage = chatMessages.lastElementChild;
            if (lastMessage) {
                lastMessage.scrollIntoView({
                    behavior: 'smooth',
                    block: 'end',
                    inline: 'nearest'
                });
            }
        }, 100);

        return messageDiv;
    }

    async function generatePPT(content) {
        try {
            const response = await fetch(getBasePath() + '/generate-ppt', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ content }),
                credentials: 'same-origin'
            });
            
            // Check for 401 Unauthorized
            if (response.status === 401) {
                // Redirect to login page
                window.location.href = getBasePath() + '/login';
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            // Create temporary link to download PPT
            const link = document.createElement('a');
            link.href = data.download_url;
            link.download = 'presentation.pptx';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
        } catch (error) {
            console.error('Error generating PPT:', error);
            alert('Error generating presentation');
        }
    }
    
    if (mcqButton && mcqSection) {
        mcqButton.addEventListener('click', () => {
            chatSection.classList.remove('active');
            documentsSection.classList.remove('active');
            mcqSection.classList.add('active');
            
            chatButton.classList.remove('active');
            filesButton.classList.remove('active');
            mcqButton.classList.add('active');
            
            document.getElementById('mainHeader').style.display = 'none'; // Hide header
            document.querySelector('.input-area').style.display = 'none'; // Hide input area
        });
        
        // Connect button handlers to the global functions from mcq.js
        document.getElementById('mcqSubjectSubmit').addEventListener('click', function() {
            const subject = document.getElementById('mcqSubject').value.trim();
            if (!subject) {
                alert('Please enter a subject');
                return;
            }
            window.startMcqTest(subject);
        });
        
        document.getElementById('submitMcqButton').addEventListener('click', function() {
            window.submitMcqTest();
        });
        
        document.getElementById('resetMcqButton').addEventListener('click', function() {
            window.resetMcqTest();
        });
    }
    
    // Subject selection
    document.querySelectorAll('.subject-pill').forEach(pill => {
        pill.addEventListener('click', function() {
            // Toggle active class
            document.querySelectorAll('.subject-pill').forEach(p => p.classList.remove('active'));
            this.classList.add('active');
            
            // Update content type based on subject
            const subject = this.dataset.subject.toLowerCase();
            if (subject === 'mathematics') {
                contentTypeSelector.value = 'math';
            } else if (subject === 'literature') {
                contentTypeSelector.value = 'literature';
            } else if (['physics', 'biology', 'chemistry'].includes(subject)) {
                contentTypeSelector.value = 'science';
            } else {
                contentTypeSelector.value = 'general';
            }
            
            // Add subject-specific prompts to input placeholder
            const subjectPrompts = {
                'mathematics': 'Solve this equation: 3x + 5 = 17',
                'physics': 'Explain the concept of gravitational potential energy',
                'computer science': 'How does a binary search algorithm work?',
                'literature': 'Analyze the themes in Hamlet by Shakespeare',
                'history': 'What were the causes of World War I?',
                'economics': 'Explain the concept of price elasticity',
                'biology': 'Describe the process of cellular respiration',
                'chemistry': 'How do ionic and covalent bonds differ?'
            };
            
            if (subjectPrompts[subject]) {
                messageInput.placeholder = `Try asking: "${subjectPrompts[subject]}"`;
            }
        });
    });

    // Global variables for MCQ test
    let currentMcqTest = null;
    let userMcqAnswers = {};

    // MCQ Test Functionality
    async function startMcqTest(subject) {
        try {
            // Show loading overlay
            showLoading('Generating MCQ test...');
            
            const numQuestions = document.getElementById('numQuestions').value;
            
            const response = await fetch(getBasePath() + '/generate-mcq', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    subject: subject,
                    num_questions: numQuestions
                }),
                credentials: 'same-origin'
            });
            
            // Check for unauthorized
            if (response.status === 401) {
                hideLoading();
                window.location.href = getBasePath() + '/login';
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            hideLoading();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            if (!data.mcq_test || data.mcq_test.length === 0) {
                throw new Error('No questions were generated.');
            }
            
            // Store current test
            currentMcqTest = data.mcq_test;
            userMcqAnswers = {};
            
            // Display MCQ test
            document.getElementById('mcqTestSubject').textContent = `${subject} Test`;
            displayMcqQuestions(data.mcq_test);
            
            // Show test area
            document.getElementById('mcqTestArea').style.display = 'block';
            document.getElementById('mcqResultsArea').style.display = 'none';
            document.querySelector('.mcq-form').style.display = 'none';
            
        } catch (error) {
            hideLoading();
            alert('Error generating MCQ test: ' + error.message);
            console.error('API Error:', error);
        }
    }

    function displayMcqQuestions(questions) {
        const questionsContainer = document.getElementById('mcqQuestions');
        questionsContainer.innerHTML = '';
        
        questions.forEach((question, index) => {
            const questionDiv = document.createElement('div');
            questionDiv.className = 'mcq-question';
            questionDiv.innerHTML = `
                <div class="mcq-question-text">${index + 1}. ${question.question}</div>
                <div class="mcq-options" id="options-${question.id}">
                    ${question.options.map((option, optIdx) => `
                        <div class="mcq-option" onclick="selectMcqOption('${question.id}', '${String.fromCharCode(65 + optIdx)}', this)">
                            <div class="mcq-option-letter">${String.fromCharCode(65 + optIdx)}</div>
                            <div class="mcq-option-text">${option}</div>
                        </div>
                    `).join('')}
                </div>
            `;
            
            questionsContainer.appendChild(questionDiv);
        });
        
        // Render any math content
        renderMathContent();
    }

    // Submit MCQ test
    async function submitMcqTest() {
        if (!currentMcqTest) {
            alert('No active test to submit.');
            return;
        }
        
        const totalQuestions = currentMcqTest.length;
        const answeredQuestions = Object.keys(userMcqAnswers).length;
        
        if (answeredQuestions < totalQuestions) {
            if (!confirm(`You've only answered ${answeredQuestions} out of ${totalQuestions} questions. Are you sure you want to submit?`)) {
                return;
            }
        }
        
        try {
            showLoading('Evaluating test...');
            
            const response = await fetch(getBasePath() + '/evaluate-mcq', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mcq_test: currentMcqTest,
                    user_answers: userMcqAnswers
                }),
                credentials: 'same-origin'
            });
            
            if (response.status === 401) {
                hideLoading();
                window.location.href = getBasePath() + '/login';
                return;
            }
            
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            
            const data = await response.json();
            hideLoading();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            showMcqResults(data.results);
            
        } catch (error) {
            hideLoading();
            alert('Error evaluating test: ' + error.message);
            console.error('API Error:', error);
        }
    }

    // Show MCQ test results
    function showMcqResults(results) {
        document.getElementById('mcqTestArea').style.display = 'none';
        document.getElementById('mcqResultsArea').style.display = 'block';
        
        // Update score
        const scorePercentage = Math.round(results.percentage);
        document.getElementById('scorePercentage').textContent = `${scorePercentage}%`;
        document.getElementById('scoreMessage').textContent = getScoreMessage(scorePercentage);
        
        // Create chart
        createResultChart(results);
        
        // Show detailed results
        const detailsContainer = document.getElementById('mcqResultDetails');
        detailsContainer.innerHTML = '<h3>Question Analysis</h3>';
        
        results.details.forEach(detail => {
            const question = currentMcqTest.find(q => q.id === detail.question_id);
            
            if (!question) return;
            
            const questionDiv = document.createElement('div');
            questionDiv.className = 'mcq-result-question';
            
            let statusIcon, statusClass;
            if (detail.correct) {
                statusIcon = '✓';
                statusClass = 'correct';
            } else {
                statusIcon = '✗';
                statusClass = 'incorrect';
            }
            
            questionDiv.innerHTML = `
                <div class="mcq-question-text ${statusClass}">
                    <span class="result-status">${statusIcon}</span> ${question.question}
                </div>
                <div class="mcq-options">
                    ${question.options.map((option, idx) => {
                        const optionLetter = String.fromCharCode(65 + idx);
                        let optionClass = '';
                        
                        if (optionLetter === question.correct_answer) {
                            optionClass = 'correct';
                        } else if (optionLetter === detail.user_answer && optionLetter !== question.correct_answer) {
                            optionClass = 'incorrect';
                        }
                        
                        return `
                            <div class="mcq-option ${optionClass}">
                                <div class="mcq-option-letter">${optionLetter}</div>
                                <div class="mcq-option-text">${option}</div>
                            </div>
                        `;
                    }).join('')}
                </div>
                <div class="mcq-explanation">
                    <strong>Explanation:</strong> ${question.explanation}
                </div>
            `;
            
            detailsContainer.appendChild(questionDiv);
        });
        
        // Add event listener to new test button
        document.getElementById('newMcqButton').addEventListener('click', resetMcqTest);
        
        // Render any math content in the explanations
        renderMathContent();
    }

    function createResultChart(results) {
        const ctx = document.getElementById('mcqResultChart').getContext('2d');
        
        // Clear any existing chart
        if (window.resultChart) {
            window.resultChart.destroy();
        }
        
        // Calculate percentages
        const correctPercentage = results.percentage;
        const incorrectPercentage = 100 - correctPercentage;
        
        // Create pie chart with Chart.js
        window.resultChart = new Chart(ctx, {
            type: 'pie',
            data: {
                labels: ['Correct', 'Incorrect'],
                datasets: [{
                    data: [correctPercentage, incorrectPercentage],
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.7)',  // Green for correct
                        'rgba(239, 68, 68, 0.7)'    // Red for incorrect
                    ],
                    borderColor: [
                        'rgba(16, 185, 129, 1)',
                        'rgba(239, 68, 68, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.raw || 0;
                                return `${label}: ${value.toFixed(1)}%`;
                            }
                        }
                    }
                }
            }
        });
    }

    function getScoreMessage(percentage) {
        if (percentage >= 90) {
            return "Excellent! You've mastered this topic.";
        } else if (percentage >= 80) {
            return "Great job! You have a solid understanding.";
        } else if (percentage >= 70) {
            return "Good work! Keep reinforcing these concepts.";
        } else if (percentage >= 60) {
            return "You're on the right track. Review the areas you missed.";
        } else {
            return "Keep studying! Focus on the areas you missed.";
        }
    }

    function resetMcqTest() {
        // Clear current test data
        currentMcqTest = null;
        userMcqAnswers = {};
        
        // Reset UI
        document.getElementById('mcqQuestions').innerHTML = '';
        document.getElementById('mcqTestArea').style.display = 'none';
        document.getElementById('mcqResultsArea').style.display = 'none';
        document.querySelector('.mcq-form').style.display = 'block';
        
        // Clear subject input
        document.getElementById('mcqSubject').value = '';
    }

    // Add math rendering functionality
    function renderMathContent() {
        // Check if MathJax is loaded
        if (typeof MathJax !== 'undefined') {
            MathJax.typesetPromise().then(() => {
                console.log('Math content rendered');
            }).catch(err => {
                console.error('Error rendering math:', err);
            });
        }
    }

    // Make renderMathContent available globally for use in mcq.js
    window.renderMathContent = renderMathContent;

    // Bind event listeners
    if (sendButton) {
        sendButton.addEventListener('click', sendMessage);
    }
    
    if (messageInput) {
        messageInput.addEventListener('keydown', function(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        });
    }

    if (dropzone) {
        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            handleFiles(e.dataTransfer.files);
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            handleFiles(e.target.files);
        });
    }

    // History sidebar functionality
    const historyButton = document.getElementById('historyButton');
    const historySidebar = document.getElementById('historySidebar');
    const closeHistoryBtn = document.getElementById('closeHistoryBtn');

    if (historyButton && historySidebar) {
        historyButton.addEventListener('click', () => {
            historySidebar.classList.toggle('open');
            updateHistorySidebar();
        });

        if (closeHistoryBtn) {
            closeHistoryBtn.addEventListener('click', () => {
                historySidebar.classList.remove('open');
            });
        }
    }
    
    // Clear memory button functionality
    const clearMemoryButton = document.getElementById('clearMemoryButton');
    if (clearMemoryButton) {
        clearMemoryButton.addEventListener('click', async () => {
            if (!confirm('Are you sure you want to clear the conversation memory? This will reset the AI\'s memory of your current conversation.')) {
                return;
            }
            
            try {
                const response = await fetch(getBasePath() + '/clear-memory', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    credentials: 'same-origin'
                });
                
                if (!response.ok) {
                    throw new Error('Failed to clear memory');
                }
                
                const toast = document.createElement('div');
                toast.className = 'toast-message success';
                toast.textContent = 'Conversation memory cleared';
                document.body.appendChild(toast);
                
                setTimeout(() => {
                    toast.classList.add('fade-out');
                    setTimeout(() => toast.remove(), 300);
                }, 3000);
                
            } catch (error) {
                console.error('Error clearing memory:', error);
                alert('Failed to clear memory: ' + error.message);
            }
        });
    }
    
    // New chat button functionality
    const newChatButton = document.getElementById('newChatButton');
    if (newChatButton) {
        newChatButton.addEventListener('click', () => {
            startNewChat();
        });
    }

    // Initialize the application
    async function initialize() {
        try {
            // Initialize chat ID for new session
            if (!currentChatId) {
                currentChatId = generateChatId();
            }
            
            // Update files list if we're on the documents page
            if (document.getElementById('loadedFiles')) {
                await updateFilesList();
            }
            
            // Update history sidebar
            updateHistorySidebar();
            
        } catch (error) {
            console.error('Error during initialization:', error);
        }
    }
    
    // Call initialize function
    initialize();
});
