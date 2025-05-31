// Chat state
let chatHistory = [];

// Utility functions
function getChatApiUrl() {
    return document.getElementById('chatApiUrl').value.trim();
}

function getChatProjectId() {
    return document.getElementById('chatProjectId').value.trim();
}

function getChatLimit() {
    return parseInt(document.getElementById('chatLimit').value) || 5;
}

function showChatLoading() {
    document.getElementById('chatLoading').classList.remove('hidden');
}

function hideChatLoading() {
    document.getElementById('chatLoading').classList.add('hidden');
}

function scrollToBottom() {
    const chatMessages = document.getElementById('chatMessages');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// Add message to chat
function addMessage(text, isUser = false, isError = false) {
    const chatMessages = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    
    if (isUser) {
        messageDiv.innerHTML = `
            <div class="flex items-start space-x-3 justify-end">
                <div class="bg-bright-green text-white rounded-lg p-3 max-w-3xl">
                    <p>${text}</p>
                </div>
                <div class="w-8 h-8 bg-medium-green rounded-full flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-user text-white text-sm"></i>
                </div>
            </div>
        `;
    } else {
        const bgColor = isError ? 'bg-red-100 border border-red-300' : 'bg-gray-100';
        const textColor = isError ? 'text-red-800' : 'text-gray-800';
        
        messageDiv.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="w-8 h-8 bg-bright-green rounded-full flex items-center justify-center flex-shrink-0">
                    <i class="fas fa-robot text-dark-green text-sm"></i>
                </div>
                <div class="${bgColor} rounded-lg p-3 max-w-3xl">
                    <p class="${textColor}">${text}</p>
                </div>
            </div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

// Add typing indicator
function addTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    
    typingDiv.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="w-8 h-8 bg-bright-green rounded-full flex items-center justify-center flex-shrink-0">
                <i class="fas fa-robot text-dark-green text-sm"></i>
            </div>
            <div class="bg-gray-100 rounded-lg p-3">
                <div class="flex space-x-1">
                    <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                    <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.1s"></div>
                    <div class="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style="animation-delay: 0.2s"></div>
                </div>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    scrollToBottom();
}

// Remove typing indicator
function removeTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

// Send message function
async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    const message = chatInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // Disable input and button
    chatInput.disabled = true;
    sendButton.disabled = true;
    sendButton.innerHTML = '<i class="fas fa-spinner fa-spin mr-2"></i>Sending...';
    
    // Add user message to chat
    addMessage(message, true);
    
    // Clear input
    chatInput.value = '';
    
    // Add typing indicator
    addTypingIndicator();
    
    // Prepare request
    const requestBody = {
        text: message,
        limit: getChatLimit()
    };
    
    try {
        const response = await fetch(`${getChatApiUrl()}/api/v1/nlp/index/answer/${getChatProjectId()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        // Remove typing indicator
        removeTypingIndicator();
        
        if (response.ok) {
            if (result.answer) {
                addMessage(result.answer);
                
                // Store in chat history if needed
                chatHistory.push({
                    user: message,
                    assistant: result.answer,
                    timestamp: new Date().toISOString()
                });
            } else {
                addMessage("I'm sorry, I couldn't generate an answer for your question. Please try rephrasing it or make sure your documents are properly indexed.", false, true);
            }
        } else {
            let errorMessage = "I'm sorry, there was an error processing your request.";
            
            if (result.signal) {
                switch (result.signal) {
                    case 'project_not_found':
                        errorMessage = "Project not found. Please check your project ID.";
                        break;
                    case 'vectordb_search_error':
                        errorMessage = "Search error. Please make sure your documents are indexed.";
                        break;
                    case 'rag_answer_error':
                        errorMessage = "Error generating answer. Please try again.";
                        break;
                    default:
                        errorMessage = `Error: ${result.signal}`;
                }
            }
            
            addMessage(errorMessage, false, true);
        }
    } catch (error) {
        removeTypingIndicator();
        addMessage(`Network error: ${error.message}. Please check if the server is running and the API URL is correct.`, false, true);
    } finally {
        // Re-enable input and button
        chatInput.disabled = false;
        sendButton.disabled = false;
        sendButton.innerHTML = '<i class="fas fa-paper-plane mr-2"></i>Send';
        chatInput.focus();
    }
}

// Clear chat function
function clearChat() {
    const chatMessages = document.getElementById('chatMessages');
    
    // Keep only the welcome message
    chatMessages.innerHTML = `
        <div class="flex items-start space-x-3">
            <div class="w-8 h-8 bg-bright-green rounded-full flex items-center justify-center flex-shrink-0">
                <i class="fas fa-robot text-dark-green text-sm"></i>
            </div>
            <div class="bg-gray-100 rounded-lg p-3 max-w-3xl">
                <p class="text-gray-800">Hello! I'm your Mini-RAG assistant. I can help you find information from your uploaded documents. Just ask me a question!</p>
            </div>
        </div>
    `;
    
    // Clear chat history
    chatHistory = [];
    
    // Focus on input
    document.getElementById('chatInput').focus();
}

// Auto-resize textarea function (if we want to make it expandable)
function autoResize(textarea) {
    textarea.style.height = 'auto';
    textarea.style.height = textarea.scrollHeight + 'px';
}

// Test API Connection Function
async function testChatApi() {
    try {
        const response = await fetch(`${getChatApiUrl()}/api/v1/welcome`, {
            method: 'GET'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            addMessage(`✅ API Connection Successful! 
                       Version: ${result.version}
                       Available endpoints: ${Object.keys(result.endpoints).length}
                       Server is running properly.`, false, false);
        } else {
            addMessage(`❌ API Test failed: ${result.signal || 'Unknown error'}`, false, true);
        }
    } catch (error) {
        addMessage(`❌ Network error: ${error.message}. Please check if the server is running and the API URL is correct.`, false, true);
    }
}

// Initialize chatbot
document.addEventListener('DOMContentLoaded', function() {
    // Focus on input when page loads
    document.getElementById('chatInput').focus();
    
    // Add event listener for Ctrl+Enter to send message
    document.getElementById('chatInput').addEventListener('keydown', function(e) {
        if (e.ctrlKey && e.key === 'Enter') {
            sendMessage();
        }
    });
    
    // Add event listener for settings changes
    document.getElementById('chatProjectId').addEventListener('change', function() {
        addMessage("Project ID changed. You may need to re-index your documents for this project.", false, false);
    });
    
    // Prevent form submission on Enter in settings
    ['chatProjectId', 'chatApiUrl', 'chatLimit'].forEach(id => {
        document.getElementById(id).addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                document.getElementById('chatInput').focus();
            }
        });
    });
    
    console.log('Mini-RAG Chatbot loaded successfully!');
}); 