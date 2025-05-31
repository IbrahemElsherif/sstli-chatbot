// Utility functions
function getApiUrl() {
    return document.getElementById('apiUrl').value.trim();
}

function getProjectId() {
    return document.getElementById('projectId').value.trim();
}

function showLoading() {
    document.getElementById('loadingModal').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingModal').classList.add('hidden');
}

function showResult(elementId, type, message) {
    const element = document.getElementById(elementId);
    element.className = `mt-4 p-4 rounded-lg ${type === 'success' ? 'bg-green-100 border border-green-400 text-green-700' : 'bg-red-100 border border-red-400 text-red-700'}`;
    element.innerHTML = `
        <div class="flex items-center">
            <i class="fas ${type === 'success' ? 'fa-check-circle' : 'fa-exclamation-circle'} mr-2"></i>
            <span>${message}</span>
        </div>
    `;
    element.classList.remove('hidden');
}

// File Upload Function
async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showResult('uploadResult', 'error', 'Please select a file to upload.');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/data/upload/${getProjectId()}`, {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('uploadResult', 'success', `File uploaded successfully! File ID: ${result.file_id}`);
            // Store the file ID for later use
            document.getElementById('fileId').value = result.file_id;
        } else {
            showResult('uploadResult', 'error', `Upload failed: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        showResult('uploadResult', 'error', `Network error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Process Files Function
async function processFiles() {
    const fileId = document.getElementById('fileId').value.trim();
    const chunkSize = parseInt(document.getElementById('chunkSize').value) || 1000;
    const overlapSize = parseInt(document.getElementById('overlapSize').value) || 150;
    const doReset = document.getElementById('doReset').checked ? 1 : 0;
    
    // Build request body - only include file_id if it has a value
    const requestBody = {
        chunk_size: chunkSize,
        overlap_size: overlapSize,
        do_reset: doReset
    };
    
    // Only add file_id if it's not empty
    if (fileId) {
        requestBody.file_id = fileId;
    }
    
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/data/process/${getProjectId()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('processResult', 'success', 
                `Processing completed! Inserted ${result.inserted_chunks} chunks from ${result.processed_files} files.`);
        } else {
            showResult('processResult', 'error', `Processing failed: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        showResult('processResult', 'error', `Network error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Push to Index Function
async function pushToIndex() {
    const doReset = document.getElementById('resetIndex').checked ? 1 : 0;
    
    const requestBody = {
        do_reset: doReset
    };
    
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/nlp/index/push/${getProjectId()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('indexResult', 'success', 
                `Indexing completed! ${result.inserted_items_count} items pushed to vector database.`);
        } else {
            showResult('indexResult', 'error', `Indexing failed: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        showResult('indexResult', 'error', `Network error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Get Index Info Function
async function getIndexInfo() {
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/nlp/index/info/${getProjectId()}`, {
            method: 'GET'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const infoElement = document.getElementById('indexInfo');
            infoElement.innerHTML = `
                <h4 class="font-bold text-dark-green mb-2">Vector Database Collection Info</h4>
                <div class="space-y-2">
                    <div><strong>Status:</strong> ${result.collection_info ? 'Available' : 'Not Available'}</div>
                    ${result.collection_info ? `
                        <div><strong>Points Count:</strong> ${result.collection_info.points_count || 'N/A'}</div>
                        <div><strong>Vectors Count:</strong> ${result.collection_info.vectors_count || 'N/A'}</div>
                        <div><strong>Index Status:</strong> ${result.collection_info.status || 'N/A'}</div>
                    ` : ''}
                </div>
            `;
            infoElement.classList.remove('hidden');
        } else {
            showResult('indexInfo', 'error', `Failed to get index info: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        const infoElement = document.getElementById('indexInfo');
        infoElement.innerHTML = `<div class="text-red-600">Network error: ${error.message}</div>`;
        infoElement.classList.remove('hidden');
    } finally {
        hideLoading();
    }
}

// Search Index Function
async function searchIndex() {
    const query = document.getElementById('searchQuery').value.trim();
    const limit = parseInt(document.getElementById('searchLimit').value) || 5;
    
    if (!query) {
        showResult('searchResults', 'error', 'Please enter a search query.');
        return;
    }
    
    const requestBody = {
        text: query,
        limit: limit
    };
    
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/nlp/index/search/${getProjectId()}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        
        if (response.ok) {
            const resultsElement = document.getElementById('searchResults');
            
            if (result.results && result.results.length > 0) {
                resultsElement.innerHTML = `
                    <div class="bg-green-100 border border-green-400 text-green-700 p-4 rounded-lg">
                        <h4 class="font-bold mb-3">Search Results (${result.results.length} found)</h4>
                        <div class="space-y-3">
                            ${result.results.map((item, index) => `
                                <div class="bg-white p-3 rounded border-l-4 border-bright-green">
                                    <div class="flex justify-between items-start mb-2">
                                        <span class="font-semibold text-sm text-gray-600">Result ${index + 1}</span>
                                        <span class="text-sm text-gray-500">Score: ${item.score ? item.score.toFixed(4) : 'N/A'}</span>
                                    </div>
                                    <p class="text-gray-800">${item.text || 'No text available'}</p>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                `;
            } else {
                resultsElement.innerHTML = `
                    <div class="bg-yellow-100 border border-yellow-400 text-yellow-700 p-4 rounded-lg">
                        <div class="flex items-center">
                            <i class="fas fa-info-circle mr-2"></i>
                            <span>No results found for your query.</span>
                        </div>
                    </div>
                `;
            }
            
            resultsElement.classList.remove('hidden');
        } else {
            showResult('searchResults', 'error', `Search failed: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        showResult('searchResults', 'error', `Network error: ${error.message}`);
    } finally {
        hideLoading();
    }
}

// Test API Connection Function
async function testApi() {
    showLoading();
    
    try {
        const response = await fetch(`${getApiUrl()}/api/v1/welcome`, {
            method: 'GET'
        });
        
        const result = await response.json();
        
        if (response.ok) {
            showResult('apiTestResult', 'success', 
                `API Connection Successful! Version: ${result.version}. Available endpoints: ${Object.keys(result.endpoints).length}`);
        } else {
            showResult('apiTestResult', 'error', `API Test failed: ${result.signal || 'Unknown error'}`);
        }
    } catch (error) {
        showResult('apiTestResult', 'error', `Network error: ${error.message}. Please check if the server is running and the API URL is correct.`);
    } finally {
        hideLoading();
    }
}

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    // Add event listeners for Enter key on input fields
    document.getElementById('searchQuery').addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            searchIndex();
        }
    });
    
    console.log('Mini-RAG Management Dashboard loaded successfully!');
}); 