/**
 * SILVER UMBRELLA - FRONTEND SCRIPT
 * Handles URL checking, results display, and user interactions
 */

// ===================================================================
// CONFIGURATION
// ===================================================================
const CONFIG = {
    // API endpoint (adjust for production)
    API_BASE_URL: window.location.hostname === 'localhost' 
        ? 'http://localhost:5000'
        : 'https://your-production-api.com',
    
    // Animation durations (milliseconds)
    ANIMATION_DURATION: 300,
    
    // API timeout (milliseconds)
    API_TIMEOUT: 30000
};

// ===================================================================
// DOM ELEMENTS
// ===================================================================
const elements = {
    form: document.getElementById('checkerForm'),
    urlInput: document.getElementById('urlInput'),
    analyzeButton: document.getElementById('analyzeButton'),
    loadingContainer: document.getElementById('loadingContainer'),
    resultsContainer: document.getElementById('resultsContainer')
};

// ===================================================================
// STATE MANAGEMENT
// ===================================================================
const state = {
    isChecking: false,
    currentURL: ''
};

// ===================================================================
// URL VALIDATION
// ===================================================================

/**
 * Validates if a string is a properly formatted URL
 * @param {string} urlString - The URL to validate
 * @returns {boolean} - True if valid URL
 */
function isValidURL(urlString) {
    // Remove whitespace
    urlString = urlString.trim();
    
    if (!urlString) {
        return false;
    }
    
    // Add protocol if missing
    if (!urlString.match(/^https?:\/\//i)) {
        urlString = 'http://' + urlString;
    }
    
    try {
        new URL(urlString);
        return true;
    } catch (error) {
        return false;
    }
}

/**
 * Normalizes URL for API submission
 * @param {string} urlString - The URL to normalize
 * @returns {string} - Normalized URL
 */
function normalizeURL(urlString) {
    urlString = urlString.trim();
    
    if (!urlString.match(/^https?:\/\//i)) {
        urlString = 'http://' + urlString;
    }
    
    return urlString;
}

// ===================================================================
// UI STATE MANAGEMENT
// ===================================================================

/**
 * Shows the loading state
 */
function showLoading() {
    state.isChecking = true;
    elements.analyzeButton.disabled = true;
    elements.loadingContainer.hidden = false;
    elements.resultsContainer.hidden = true;
    
    // Smooth scroll to loading
    elements.loadingContainer.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

/**
 * Hides the loading state
 */
function hideLoading() {
    state.isChecking = false;
    elements.analyzeButton.disabled = false;
    elements.loadingContainer.hidden = true;
}

/**
 * Shows the results container with animation
 */
function showResults() {
    elements.resultsContainer.hidden = false;
    
    // Smooth scroll to results
    setTimeout(() => {
        elements.resultsContainer.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
}

/**
 * Hides the results container
 */
function hideResults() {
    elements.resultsContainer.hidden = true;
}

// ===================================================================
// API COMMUNICATION
// ===================================================================

/**
 * Calls the backend API to check a URL
 * @param {string} url - The URL to check
 * @returns {Promise<Object>} - API response data
 */
async function checkURL(url) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), CONFIG.API_TIMEOUT);
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/check-url`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ url: url }),
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.message || 'Unable to check this website');
        }
        
        return await response.json();
        
    } catch (error) {
        clearTimeout(timeoutId);
        
        if (error.name === 'AbortError') {
            throw new Error('Request took too long. Please try again.');
        }
        
        throw error;
    }
}

// ===================================================================
// RESULTS RENDERING
// ===================================================================

/**
 * Renders the safe result
 * @param {Object} data - API response data
 * @returns {string} - HTML string
 */
function renderSafeResult(data) {
    const details = data.details && data.details.length > 0
        ? `
            <div class="result-details">
                <h3 class="details-heading">Why This Website Appears Safe:</h3>
                <ul class="details-list">
                    ${data.details.map(detail => `<li>${escapeHTML(detail)}</li>`).join('')}
                </ul>
            </div>
        `
        : '';
    
    return `
        <div class="result-card safe">
            <div class="result-verdict">
                <span class="result-icon" aria-hidden="true">✅</span>
                <h2 class="result-status">This Website Looks SAFE</h2>
            </div>
            
            <p class="result-confidence">
                We are <span class="confidence-number">${data.confidence_percent}%</span> confident in this assessment
            </p>
            
            <div class="result-explanation">
                <h3 class="explanation-heading">What This Means:</h3>
                <p class="explanation-text">${escapeHTML(data.explanation)}</p>
            </div>
            
            ${details}
            
            <button type="button" class="check-another-button" onclick="resetChecker()">
                Check Another Website
            </button>
        </div>
    `;
}

/**
 * Renders the danger result
 * @param {Object} data - API response data
 * @returns {string} - HTML string
 */
function renderDangerResult(data) {
    const details = data.details && data.details.length > 0
        ? `
            <div class="result-details">
                <h3 class="details-heading">⚠️ Warning Signs We Found:</h3>
                <ul class="details-list">
                    ${data.details.map(detail => `<li>${escapeHTML(detail)}</li>`).join('')}
                </ul>
            </div>
        `
        : '';
    
    return `
        <div class="result-card danger">
            <div class="result-verdict">
                <span class="result-icon" aria-hidden="true">🚨</span>
                <h2 class="result-status">DANGER - Do Not Visit</h2>
            </div>
            
            <p class="result-confidence">
                We are <span class="confidence-number">${data.confidence_percent}%</span> confident this is dangerous
            </p>
            
            <div class="result-explanation">
                <h3 class="explanation-heading">⚠️ Important Warning:</h3>
                <p class="explanation-text">${escapeHTML(data.explanation)}</p>
            </div>
            
            ${details}
            
            <div class="result-explanation">
                <h3 class="explanation-heading">What You Should Do:</h3>
                <p class="explanation-text">
                    <strong>Do NOT click this link.</strong> Do not enter any passwords, credit card numbers, 
                    or personal information. Delete the message that sent you this link. If you think this 
                    came from a real company, contact them directly using a phone number from their official website.
                </p>
            </div>
            
            <button type="button" class="check-another-button" onclick="resetChecker()">
                Check Another Website
            </button>
        </div>
    `;
}

/**
 * Renders an error message
 * @param {string} message - Error message
 * @returns {string} - HTML string
 */
function renderError(message) {
    return `
        <div class="result-card" style="background-color: #FEF3C7; border-color: #D97706;">
            <div class="result-verdict">
                <span class="result-icon" aria-hidden="true">⚠️</span>
                <h2 class="result-status" style="color: #D97706;">Unable to Check</h2>
            </div>
            
            <div class="result-explanation">
                <h3 class="explanation-heading">What Happened:</h3>
                <p class="explanation-text">${escapeHTML(message)}</p>
            </div>
            
            <div class="result-explanation">
                <h3 class="explanation-heading">What You Can Do:</h3>
                <p class="explanation-text">
                    Please check that you entered the website address correctly and try again. 
                    If the problem continues, this website might be having technical issues.
                </p>
            </div>
            
            <button type="button" class="check-another-button" onclick="resetChecker()">
                Try Again
            </button>
        </div>
    `;
}

/**
 * Displays results from API
 * @param {Object} data - API response data
 */
function displayResults(data) {
    const isSafe = data.prediction === 'legitimate';
    const html = isSafe ? renderSafeResult(data) : renderDangerResult(data);
    
    elements.resultsContainer.innerHTML = html;
    showResults();
}

/**
 * Displays error message
 * @param {string} message - Error message
 */
function displayError(message) {
    elements.resultsContainer.innerHTML = renderError(message);
    showResults();
}

// ===================================================================
// SECURITY HELPERS
// ===================================================================

/**
 * Escapes HTML to prevent XSS attacks
 * @param {string} text - Text to escape
 * @returns {string} - Escaped text
 */
function escapeHTML(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===================================================================
// MAIN FORM HANDLER
// ===================================================================

/**
 * Handles form submission
 * @param {Event} event - Form submit event
 */
async function handleFormSubmit(event) {
    event.preventDefault();
    
    // Prevent double submission
    if (state.isChecking) {
        return;
    }
    
    // Get and validate URL
    const urlValue = elements.urlInput.value.trim();
    
    if (!urlValue) {
        alert('Please enter a website address to check.');
        elements.urlInput.focus();
        return;
    }
    
    if (!isValidURL(urlValue)) {
        alert('This doesn\'t look like a valid website address.\n\nPlease check and try again.\n\nExample: google.com or https://example.com');
        elements.urlInput.focus();
        return;
    }
    
    // Normalize and store URL
    const normalizedURL = normalizeURL(urlValue);
    state.currentURL = normalizedURL;
    
    // Show loading state
    showLoading();
    
    try {
        // Call API
        const data = await checkURL(normalizedURL);
        
        // Hide loading and show results
        hideLoading();
        displayResults(data);
        
    } catch (error) {
        // Hide loading and show error
        hideLoading();
        console.error('Error checking URL:', error);
        displayError(error.message || 'Unable to check this website. Please try again.');
    }
}

/**
 * Resets the checker to initial state
 */
function resetChecker() {
    elements.urlInput.value = '';
    elements.urlInput.focus();
    hideResults();
    hideLoading();
    
    // Smooth scroll back to input
    elements.urlInput.scrollIntoView({ 
        behavior: 'smooth', 
        block: 'center' 
    });
}

// ===================================================================
// EVENT LISTENERS
// ===================================================================

// Form submission
elements.form.addEventListener('submit', handleFormSubmit);

// Input validation on blur
elements.urlInput.addEventListener('blur', function() {
    const value = this.value.trim();
    if (value && !isValidURL(value)) {
        this.setAttribute('aria-invalid', 'true');
    } else {
        this.removeAttribute('aria-invalid');
    }
});

// Clear validation on input
elements.urlInput.addEventListener('input', function() {
    this.removeAttribute('aria-invalid');
});

// ===================================================================
// INITIALIZATION
// ===================================================================

/**
 * Initializes the application
 */
function init() {
    console.log('Silver Umbrella initialized');
    
    // Focus on input field
    elements.urlInput.focus();
    
    // Test API connection
    testAPIConnection();
}

/**
 * Tests connection to backend API
 */
async function testAPIConnection() {
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/health`);
        if (response.ok) {
            console.log('✅ API connection successful');
        } else {
            console.warn('⚠️ API health check returned non-OK status');
        }
    } catch (error) {
        console.error('❌ Cannot connect to API:', error);
        console.log('Make sure backend is running at:', CONFIG.API_BASE_URL);
    }
}

// ===================================================================
// START APPLICATION
// ===================================================================

// Run initialization when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

// Make resetChecker available globally for onclick handlers
window.resetChecker = resetChecker;