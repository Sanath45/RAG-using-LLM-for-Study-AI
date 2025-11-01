/**
 * MCQ Test Module - Handles test generation, display and evaluation
 */

// Global variables for MCQ test
let currentMcqTest = null;
let userMcqAnswers = {};

/**
 * Gets base path for API calls
 * @returns {string} Base URL
 */
function getBasePath() {
    const origin = window.location.origin;
    let appPath = '';
    const pathnameParts = window.location.pathname.split('/');
    
    if (pathnameParts.length > 0) {
        if (pathnameParts[pathnameParts.length - 1].includes('.') || 
            pathnameParts[pathnameParts.length - 1] === '') {
            pathnameParts.pop();
        }
        appPath = pathnameParts.join('/');
    }
    
    return origin + appPath;
}

/**
 * Initiates MCQ test generation by sending request to the server
 * @param {string} subject - The academic subject for the test
 */
async function startMcqTest(subject) {
    try {
        // Ensure we're in the MCQ tab
        ensureMcqTabActive();
        
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
        showToast('Error generating MCQ test: ' + error.message, 'error');
        console.error('API Error:', error);
    }
}

/**
 * Make sure MCQ tab is active when displaying MCQ content
 */
function ensureMcqTabActive() {
    // Get references to sections and navigation buttons
    const mcqSection = document.getElementById('mcqSection');
    const chatSection = document.getElementById('chatSection');
    const documentsSection = document.getElementById('documentsSection');
    
    const mcqButton = document.getElementById('mcqButton');
    const chatButton = document.getElementById('chatButton');
    const filesButton = document.getElementById('filesButton');
    
    // Set the correct active classes
    mcqSection.classList.add('active');
    chatSection.classList.remove('active');
    documentsSection.classList.remove('active');
    
    mcqButton.classList.add('active');
    chatButton.classList.remove('active');
    filesButton.classList.remove('active');
    
    // Hide header and input area
    document.getElementById('mainHeader').style.display = 'none';
    document.querySelector('.input-area').style.display = 'none';
}

/**
 * Displays MCQ questions in the UI
 * @param {Array} questions - Array of question objects
 */
function displayMcqQuestions(questions) {
    const questionsContainer = document.getElementById('mcqQuestions');
    questionsContainer.innerHTML = '';
    
    questions.forEach((question, index) => {
        const questionDiv = document.createElement('div');
        questionDiv.className = 'mcq-question';
        questionDiv.innerHTML = `
            <div class="mcq-question-text">${index + 1}. ${question.question}</div>
            <div class="mcq-options" id="options-${question.id}">
                ${question.options.map((option, optIdx) => {
                    const optionLetter = String.fromCharCode(65 + optIdx);
                    return `
                        <div class="mcq-option" onclick="selectMcqOption('${question.id}', '${optionLetter}', this)">
                            <div class="mcq-option-letter">${optionLetter}</div>
                            <div class="mcq-option-text">${option}</div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
        
        questionsContainer.appendChild(questionDiv);
    });
    
    // Render any math content
    renderMathContent();
}

/**
 * Handles MCQ option selection
 * @param {string|number} questionId - Question identifier
 * @param {string} selectedOption - Selected option letter (A, B, C, D)
 * @param {HTMLElement} element - Selected option DOM element
 */
function selectMcqOption(questionId, selectedOption, element) {
    // Clear previously selected option in this question group
    const optionsContainer = document.getElementById(`options-${questionId}`);
    if (optionsContainer) {
        const options = optionsContainer.querySelectorAll('.mcq-option');
        options.forEach(option => option.classList.remove('selected'));
    }
    
    // Mark this option as selected
    element.classList.add('selected');
    
    // Store user's answer
    userMcqAnswers[questionId] = selectedOption;
}

/**
 * Submit MCQ test for evaluation
 */
async function submitMcqTest() {
    if (!currentMcqTest) {
        showToast('No active test to submit.', 'error');
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
        // Ensure we're in the MCQ tab
        ensureMcqTabActive();
        
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
        showToast('Error evaluating test: ' + error.message, 'error');
        console.error('API Error:', error);
    }
}

/**
 * Displays MCQ test results
 * @param {Object} results - Test results with score and details
 */
function showMcqResults(results) {
    // Ensure we're in the MCQ tab
    ensureMcqTabActive();
    
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
                <strong>Explanation:</strong> ${question.explanation || 'No explanation provided.'}
            </div>
        `;
        
        detailsContainer.appendChild(questionDiv);
    });
    
    // Add event listener to new test button
    document.getElementById('newMcqButton').addEventListener('click', resetMcqTest);
    
    // Render any math content in the explanations
    renderMathContent();
}

/**
 * Creates a chart showing test results
 * @param {Object} results - Test results with score data
 */
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

/**
 * Gets motivational message based on score percentage
 * @param {number} percentage - Score percentage
 * @returns {string} Motivational message
 */
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

/**
 * Resets the MCQ test state
 */
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

/**
 * Renders math content using MathJax
 */
function renderMathContent() {
    // Use window.renderMathContent if available (from script.js), otherwise implement locally
    if (typeof window.renderMathContent === 'function') {
        window.renderMathContent();
    } else if (typeof MathJax !== 'undefined') {
        MathJax.typesetPromise().then(() => {
            console.log('Math content rendered');
        }).catch(err => {
            console.error('Error rendering math:', err);
        });
    }
}

/**
 * Shows a loading overlay with message
 * @param {string} message - Message to display
 */
function showLoading(message) {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        document.getElementById('loadingText').textContent = message;
        loadingOverlay.style.display = 'flex';
    }
}

/**
 * Hides the loading overlay
 */
function hideLoading() {
    const loadingOverlay = document.getElementById('loadingOverlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = 'none';
    }
}

/**
 * Shows a toast message
 * @param {string} message - Message to display
 * @param {string} type - Message type (success/error)
 */
function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast-message ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.classList.add('fade-out');
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Make functions globally available
window.startMcqTest = startMcqTest;
window.selectMcqOption = selectMcqOption;
window.submitMcqTest = submitMcqTest;
window.resetMcqTest = resetMcqTest;

// Initialize event listeners when document is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Ensure "New Test" button in results is properly connected
    const newMcqButton = document.getElementById('newMcqButton');
    if (newMcqButton) {
        newMcqButton.addEventListener('click', resetMcqTest);
    }
});
