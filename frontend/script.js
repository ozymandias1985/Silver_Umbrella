const CONFIG = {
    API_BASE_URL: 'http://localhost:5000'
};

const urlInput = document.getElementById('urlInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const loading = document.getElementById('loading');
const results = document.getElementById('results');

analyzeBtn.addEventListener('click', analyzeURL);
urlInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') analyzeURL();
});

async function analyzeURL() {
    const url = urlInput.value.trim();
    
    if (!url) {
        alert('Please enter a URL');
        return;
    }
    
    // Show loading process
    loading.classList.remove('hidden');
    results.classList.add('hidden');
    analyzeBtn.disabled = true;
    
    try {
        const response = await fetch(`${CONFIG.API_BASE_URL}/api/check-url`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ url })
        });
        
        const data = await response.json();
        showResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        showError();
    } finally {
        loading.classList.add('hidden');
        analyzeBtn.disabled = false;
    }
}

function showResults(data) {
    const isSafe = data.prediction === 'legitimate';
    
    results.className = 'results';
    results.classList.add(isSafe ? 'result-safe' : 'result-danger');
    
    results.innerHTML = `
        <div class="result-header">
            <div class="result-emoji">${data.emoji}</div>
            <div>
                <div class="result-verdict ${isSafe ? 'verdict-safe' : 'verdict-danger'}">
                    ${data.verdict}
                </div>
                <div class="result-confidence">
                    Confidence: ${data.confidence_percent}%
                </div>
            </div>
        </div>
        
        <div class="result-text">
            ${data.explanation}
        </div>
        
        ${data.details && data.details.length > 0 ? `
            <div class="result-details">
                <div class="details-title">${isSafe ? 'Good Signs:' : 'Warning Signs:'}</div>
                <ul class="details-list">
                    ${data.details.map(d => `<li>${d}</li>`).join('')}
                </ul>
            </div>
        ` : ''}
        
        <button class="check-another" onclick="checkAnother()">
            Check Another URL
        </button>
    `;
    
    results.classList.remove('hidden');
}

function showError() {
    results.className = 'results result-danger';
    results.innerHTML = `
        <div class="result-header">
            <div class="result-emoji">⚠️</div>
            <div>
                <div class="result-verdict verdict-danger">Error</div>
            </div>
        </div>
        <div class="result-text">
            Unable to analyze. Please check your connection and try again.
        </div>
        <button class="check-another" onclick="checkAnother()">Try Again</button>
    `;
    results.classList.remove('hidden');
}

function checkAnother() {
    urlInput.value = '';
    results.classList.add('hidden');
    urlInput.focus();
}

window.addEventListener('load', () => urlInput.focus());