/**
 * Semantic Image Search - Frontend JavaScript
 */

// API endpoints
const API = {
    byImage: '/api/search/by-image',
    byUrl: '/api/search/by-url',
    status: '/api/status'
};

// DOM elements initialization on document load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize UI elements
    initializeUI();
    
    // Check API status
    checkApiStatus();
    
    // Set up search functionality
    setupSearchHandlers();
});

/**
 * Initialize UI elements and tabs
 */
function initializeUI() {
    // Tabs functionality
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('border-indigo-500', 'text-indigo-600'));
            tabs.forEach(t => t.classList.add('border-transparent', 'text-gray-500'));
            tab.classList.remove('border-transparent', 'text-gray-500');
            tab.classList.add('border-indigo-500', 'text-indigo-600');
            
            tabContents.forEach(content => content.classList.add('hidden'));
            const target = tab.id.replace('tab-', '');
            document.getElementById(`${target}-section`).classList.remove('hidden');
            
            // Clear previous inputs when switching tabs
            if (target === 'upload' && window.clearCurrentFile) {
                // Don't clear when switching TO upload tab
            } else if (target === 'url') {
                // Clear URL input when switching to URL tab
                const urlInput = document.getElementById('image-url');
                const urlPreview = document.getElementById('url-preview-image');
                if (urlInput) urlInput.value = '';
                if (urlPreview) urlPreview.style.display = 'none';
            }
        });
    });
    
    // File upload functionality
    setupFileUpload();
    
    // Image URL preview
    setupUrlPreview();
}

/**
 * Setup file upload and drag-and-drop functionality
 */
function setupFileUpload() {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewImage = document.getElementById('preview-image');
    
    // Store the current file (from either file input or drag-drop)
    let currentFile = null;
    
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });
    
    fileInput.addEventListener('change', function() {
        const file = this.files[0];
        if (file) {
            currentFile = file;
            displayFile(file);
        }
    });
    
    dropArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('active');
    });
    
    dropArea.addEventListener('dragenter', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.add('active');
    });
    
    dropArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        // Only remove active class if we're leaving the drop area itself
        if (!dropArea.contains(e.relatedTarget)) {
            dropArea.classList.remove('active');
        }
    });
    
    dropArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropArea.classList.remove('active');
        
        const files = e.dataTransfer.files;
        console.log('Files dropped:', files.length);
        
        if (files.length > 0) {
            const file = files[0];
            console.log('Processing dropped file:', file.name, file.type, file.size);
            
            if (file.type.startsWith('image/')) {
                currentFile = file;
                // Update the file input with the dropped file
                try {
                    const dt = new DataTransfer();
                    dt.items.add(file);
                    fileInput.files = dt.files;
                    displayFile(file);
                    console.log('File successfully processed for upload');
                } catch (error) {
                    console.error('Error setting file input:', error);
                    // Fallback: just store the file
                    currentFile = file;
                    displayFile(file);
                }
            } else {
                alert('Please drop an image file (JPG, PNG, etc.)');
            }
        }
    });
    
    // Make currentFile accessible globally for search function
    window.getCurrentFile = function() {
        return currentFile || fileInput.files[0];
    };
    
    // Reset file function
    window.clearCurrentFile = function() {
        currentFile = null;
        fileInput.value = '';
        previewImage.style.display = 'none';
        previewImage.src = '';
    };
    
    function displayFile(file) {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImage.src = e.target.result;
                previewImage.style.display = 'block';
            }
            reader.readAsDataURL(file);
        } else {
            alert('Please select an image file');
            currentFile = null;
        }
    }
}

/**
 * Setup image URL preview functionality
 */
function setupUrlPreview() {
    const imageUrl = document.getElementById('image-url');
    const urlPreviewImage = document.getElementById('url-preview-image');
    
    imageUrl.addEventListener('input', function() {
        const url = this.value.trim();
        if (url) {
            urlPreviewImage.src = url;
            urlPreviewImage.style.display = 'block';
            urlPreviewImage.onerror = function() {
                urlPreviewImage.style.display = 'none';
                alert('Could not load image from the provided URL');
            };
        } else {
            urlPreviewImage.style.display = 'none';
        }
    });
}

/**
 * Check API status and display information
 */
function checkApiStatus() {
    fetch(API.status)
        .then(response => response.json())
        .then(data => {
            // If needed, display API status information
            console.log('API Status:', data);
        })
        .catch(error => {
            console.error('API Status Error:', error);
        });
}

/**
 * Setup search button and form handlers
 */
function setupSearchHandlers() {
    const searchBtn = document.getElementById('search-btn');
    const resultsSection = document.getElementById('results-section');
    const loadingIndicator = document.getElementById('loading-indicator');
    const noResults = document.getElementById('no-results');
    const resultsGrid = document.getElementById('results-grid');
    
    searchBtn.addEventListener('click', function() {
        // Check which search mode is active
        const isUrlSearch = !document.getElementById('url-section').classList.contains('hidden');
        
        // Get search options
        const fullMatching = document.getElementById('full-matching').checked;
        const partialMatching = document.getElementById('partial-matching').checked;
        const removeText = document.getElementById('remove-text').checked;
        const enhanceImage = document.getElementById('enhance-image').checked;
        const resultLimit = document.getElementById('result-limit').value;
        
        if (!fullMatching && !partialMatching) {
            alert('Please select at least one matching method');
            return;
        }
        
        // Handle URL search
        if (isUrlSearch) {
            const url = document.getElementById('image-url').value.trim();
            if (!url) {
                alert('Please enter an image URL');
                return;
            }
            performUrlSearch(url, fullMatching, partialMatching, removeText, enhanceImage, resultLimit);
        } 
        // Handle file upload search
        else {
            const currentFile = window.getCurrentFile && window.getCurrentFile();
            if (!currentFile) {
                alert('Please select an image to search');
                return;
            }
            performFileSearch(currentFile, fullMatching, partialMatching, removeText, enhanceImage, resultLimit);
        }
        
        // Show loading state
        resultsSection.classList.remove('hidden');
        loadingIndicator.classList.remove('hidden');
        resultsGrid.innerHTML = '';
        noResults.classList.add('hidden');
    });

    /**
     * Perform search with an uploaded file
     */
    function performFileSearch(file, fullMatching, partialMatching, removeText, enhanceImage, resultLimit) {
        console.log('Performing file search with:', {
            fileName: file.name,
            fileSize: file.size,
            fileType: file.type,
            fullMatching,
            partialMatching,
            removeText,
            enhanceImage,
            resultLimit
        });
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('use_full_matching', fullMatching);
        formData.append('use_partial_matching', partialMatching);
        formData.append('remove_text', removeText);
        formData.append('enhance_image', enhanceImage);
        formData.append('limit', resultLimit);
        
        fetch(API.byImage, {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('API response status:', response.status);
            if (!response.ok) {
                return response.text().then(text => {
                    throw new Error(`HTTP error ${response.status}: ${text}`);
                });
            }
            return response.json();
        })
        .then(data => {
            console.log('Search results:', data);
            displayResults(data.results);
            loadingIndicator.classList.add('hidden');
        })
        .catch(error => {
            console.error('Search error:', error);
            alert(`Search failed: ${error.message}`);
            loadingIndicator.classList.add('hidden');
        });
    }

    /**
     * Perform search with an image URL
     */
    function performUrlSearch(url, fullMatching, partialMatching, removeText, enhanceImage, resultLimit) {
        const searchData = {
            url: url,
            use_full_matching: fullMatching,
            use_partial_matching: partialMatching,
            remove_text: removeText,
            enhance_image: enhanceImage,
            limit: parseInt(resultLimit)
        };
        
        fetch(API.byUrl, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(searchData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            displayResults(data.results);
            loadingIndicator.classList.add('hidden');
        })
        .catch(error => {
            console.error('Search error:', error);
            alert('An error occurred during the search. Please try again.');
            loadingIndicator.classList.add('hidden');
        });
    }
}

/**
 * Display search results in the UI
 */
function displayResults(results) {
    const resultsGrid = document.getElementById('results-grid');
    const noResults = document.getElementById('no-results');
    
    resultsGrid.innerHTML = '';
    
    if (!results || results.length === 0) {
        noResults.classList.remove('hidden');
        return;
    }
    
    results.forEach(result => {
        // Get the image path from the metadata
        let imagePath = result.metadata.original_path || '';
        
        // Convert local file path to a proper URL
        if (imagePath.startsWith('/') || imagePath.includes(':\\')) {
            // Convert absolute path to relative path for serving
            const filename = imagePath.split('/').pop()?.split('\\').pop() || 'image';
            
            // Try to create a relative path from the images directory
            if (imagePath.includes('images')) {
                // Extract everything after "images/" or "images\"
                const normalizedPath = imagePath.replace(/\\/g, '/');
                const imagesIndex = normalizedPath.indexOf('images/');
                if (imagesIndex !== -1) {
                    const relativePath = normalizedPath.substring(imagesIndex + 7); // Skip "images/"
                    imagePath = `/images/${relativePath}`;  // Use direct /images/ route
                } else {
                    // Fallback to API endpoint if path structure is unexpected
                    imagePath = `/api/image/${encodeURIComponent(result.id)}`;
                }
            } else {
                // Fallback: create a proper image serving endpoint
                imagePath = `/api/image/${encodeURIComponent(result.id)}`;
            }
        }
        
        const resultCard = document.createElement('div');
        resultCard.className = 'result-card bg-white border rounded-lg overflow-hidden shadow-sm hover:shadow-md';
        
        const scorePercentage = (parseFloat(result.score) * 100).toFixed(1);
        const matchedRegions = result.matched_regions 
            ? `<span class="text-gray-500 text-sm ml-2">${result.matched_regions} matched regions</span>` 
            : '';
        
        resultCard.innerHTML = `
            <img src="${imagePath}" alt="${result.metadata.filename || 'Image result'}" class="w-full h-48 object-cover">
            <div class="p-4">
                <div class="flex justify-between items-center mb-2">
                    <h3 class="font-medium text-gray-900 truncate">${result.metadata.filename || 'Unknown'}</h3>
                    <span class="bg-indigo-100 text-indigo-800 text-xs px-2 py-1 rounded-full">${scorePercentage}%</span>
                </div>
                <div class="flex items-center">
                    <div class="h-2 flex-grow bg-gray-200 rounded-full overflow-hidden">
                        <div class="bg-indigo-600 h-full rounded-full" style="width: ${scorePercentage}%"></div>
                    </div>
                    ${matchedRegions}
                </div>
            </div>
        `;
        
        resultsGrid.appendChild(resultCard);
    });
} 