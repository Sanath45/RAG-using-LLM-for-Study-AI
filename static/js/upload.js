
document.addEventListener('DOMContentLoaded', function() {
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const textInput = document.getElementById('textInput');
    const processTextBtn = document.getElementById('processTextBtn');
    const loadingOverlay = document.getElementById('loadingOverlay');

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

    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);

        showLoading('Uploading file...');
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }

            hideLoading();
            window.location.href = '/'; // Redirect to chat interface after successful upload
        } catch (error) {
            hideLoading();
            alert(error.message || 'Error uploading file');
        }
    }

    // Text Processing
    async function processText() {
        const text = textInput.value.trim();
        if (!text) {
            alert('Please enter some text to process');
            return;
        }

        showLoading('Processing text...');
        
        try {
            const response = await fetch('/process-text', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }

            hideLoading();
            window.location.href = '/';
        } catch (error) {
            hideLoading();
            alert(error.message || 'Error processing text');
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

    // Event Listeners
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

    fileInput.addEventListener('change', (e) => handleFiles(e.target.files));
    processTextBtn.addEventListener('click', processText);
});