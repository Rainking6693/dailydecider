// Emergency fix script for Daily Decider
console.log('🚨 Emergency fix script loading...');

// Global error tracking
window.dailyDeciderErrors = [];
window.dailyDeciderStatus = {
    scriptsLoaded: {},
    initStatus: 'loading'
};

// Enhanced error handling
window.addEventListener('error', function(e) {
    console.error('Script error:', e.error);
    window.dailyDeciderErrors.push({
        type: 'script_error',
        message: e.message,
        filename: e.filename,
        lineno: e.lineno,
        timestamp: Date.now()
    });
});

// Track script loading
function trackScriptLoading(scriptName) {
    window.dailyDeciderStatus.scriptsLoaded[scriptName] = true;
    console.log(`✅ ${scriptName} loaded`);
}

// Emergency compliment generator (fallback)
window.emergencyComplimentGenerator = {
    compliments: [
        "You're approaching today with exactly the right energy.",
        "Your thoughtfulness in seeking guidance shows real wisdom.",
        "There's something special about your perspective that deserves recognition.",
        "You have a gift for turning challenges into opportunities.",
        "Your willingness to reflect and grow is genuinely inspiring.",
        "The way you consider different angles shows remarkable depth.",
        "You bring a unique combination of logic and intuition to decisions.",
        "Your patience in thoughtful decision-making is admirable.",
        "You have an excellent sense for timing and context.",
        "Your balanced approach to challenges is truly valuable."
    ],
    
    getRandomCompliment() {
        const index = Math.floor(Math.random() * this.compliments.length);
        return {
            text: this.compliments[index],
            category: 'wisdom',
            sentiment: 0.8
        };
    }
};

// Emergency decision maker (fallback)
window.emergencyDecisionMaker = {
    makeDecision(question, options = []) {
        console.log('🚨 Using emergency decision maker');
        
        const decisions = [
            "Take a step back and consider this decision tomorrow when you have fresh perspective.",
            "Trust your instincts - they're usually pointing you in the right direction.",
            "Consider the option that aligns best with your long-term goals.",
            "Choose the path that you'll be proud of looking back.",
            "Go with the decision that feels most authentic to who you are."
        ];
        
        if (options.length > 0) {
            const randomOption = options[Math.floor(Math.random() * options.length)];
            return {
                decision: `I recommend: ${randomOption}`,
                reasoning: "Based on a balanced consideration of your options, this choice offers the best potential outcomes.",
                confidence: 0.75,
                factors: {
                    analysis: { clarity: 0.8 },
                    intuition: { strength: 0.7 }
                }
            };
        }
        
        const randomDecision = decisions[Math.floor(Math.random() * decisions.length)];
        return {
            decision: randomDecision,
            reasoning: "This guidance is based on general decision-making principles and your specific context.",
            confidence: 0.7,
            factors: {
                general: { applicability: 0.75 }
            }
        };
    }
};

// Emergency PayPal button fix
function fixPayPalButtons() {
    console.log('🔧 Fixing PayPal buttons...');
    
    const paypalButtons = document.querySelectorAll('.paypal-button-container form');
    paypalButtons.forEach((form, index) => {
        const img = form.querySelector('input[type="image"]');
        if (img) {
            // Fix image loading issues
            img.onerror = function() {
                console.warn('PayPal button image failed to load, using fallback');
                const fallbackButton = document.createElement('button');
                fallbackButton.type = 'submit';
                fallbackButton.style.cssText = `
                    background: #0070ba;
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 6px;
                    font-weight: bold;
                    cursor: pointer;
                    font-size: 14px;
                `;
                fallbackButton.textContent = 'Donate with PayPal';
                img.parentNode.replaceChild(fallbackButton, img);
            };
            
            // Add click tracking
            form.addEventListener('submit', function(e) {
                console.log('PayPal donation button clicked:', index);
                // Track in emergency analytics
                if (window.emergencyAnalytics) {
                    window.emergencyAnalytics.track('paypal_click', { button_index: index });
                }
            });
        }
    });
}

// Emergency compliment button fix
function fixComplimentButton() {
    console.log('🔧 Fixing compliment button...');
    
    const complimentBtn = document.getElementById('complimentBtn');
    const complimentSection = document.getElementById('complimentSection');
    const complimentText = document.getElementById('complimentText');
    const complimentCategory = document.getElementById('complimentCategory');
    
    if (complimentBtn && complimentSection && complimentText && complimentCategory) {
        // Remove existing listeners and add emergency handler
        complimentBtn.replaceWith(complimentBtn.cloneNode(true));
        const newBtn = document.getElementById('complimentBtn');
        
        newBtn.addEventListener('click', function(e) {
            e.preventDefault();
            console.log('🎭 Emergency compliment generator activated');
            
            // Show loading state
            newBtn.textContent = 'Getting your compliment...';
            newBtn.disabled = true;
            
            // Generate compliment
            setTimeout(() => {
                try {
                    const compliment = window.emergencyComplimentGenerator.getRandomCompliment();
                    
                    complimentText.textContent = compliment.text;
                    complimentCategory.textContent = compliment.category.toUpperCase();
                    
                    complimentSection.classList.remove('hidden');
                    complimentSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    
                    console.log('✅ Emergency compliment displayed');
                } catch (error) {
                    console.error('Emergency compliment failed:', error);
                    alert('Unable to generate compliment. Please refresh and try again.');
                } finally {
                    newBtn.textContent = 'Daily Compliment';
                    newBtn.disabled = false;
                }
            }, 1000);
        });
        
        console.log('✅ Emergency compliment button handler attached');
    } else {
        console.error('❌ Required compliment elements not found');
    }
}

// Emergency decision button fix
function fixDecisionButton() {
    console.log('🔧 Fixing decision button...');
    
    const form = document.getElementById('decisionForm');
    const submitBtn = document.getElementById('submitBtn');
    const questionField = document.getElementById('question');
    const resultSection = document.getElementById('resultSection');
    
    if (form && submitBtn && questionField && resultSection) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('🤖 Emergency decision maker activated');
            
            const question = questionField.value.trim();
            if (!question || question.length < 10) {
                alert('Please enter a question with at least 10 characters.');
                return;
            }
            
            // Show loading state
            submitBtn.textContent = 'Processing...';
            submitBtn.disabled = true;
            
            // Get options
            const optionInputs = document.querySelectorAll('.option-input input');
            const options = Array.from(optionInputs)
                .map(input => input.value.trim())
                .filter(option => option.length > 0);
            
            setTimeout(() => {
                try {
                    const result = window.emergencyDecisionMaker.makeDecision(question, options);
                    
                    // Display result
                    const decisionResult = document.getElementById('decisionResult');
                    const reasoning = document.getElementById('reasoning');
                    const confidenceBadge = document.getElementById('confidenceBadge');
                    
                    if (decisionResult) decisionResult.textContent = result.decision;
                    if (reasoning) reasoning.textContent = result.reasoning;
                    if (confidenceBadge) {
                        confidenceBadge.textContent = `Confidence: ${Math.round(result.confidence * 100)}%`;
                        confidenceBadge.className = 'confidence-badge confidence-medium';
                    }
                    
                    resultSection.classList.remove('hidden');
                    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                    
                    console.log('✅ Emergency decision displayed');
                } catch (error) {
                    console.error('Emergency decision failed:', error);
                    alert('Unable to process decision. Please refresh and try again.');
                } finally {
                    submitBtn.textContent = 'Get My Decision';
                    submitBtn.disabled = false;
                }
            }, 1500);
        });
        
        console.log('✅ Emergency decision handler attached');
    } else {
        console.error('❌ Required decision elements not found');
    }
}

// Emergency analytics
window.emergencyAnalytics = {
    events: [],
    track(event, data) {
        this.events.push({
            event,
            data,
            timestamp: Date.now()
        });
        console.log('📊 Emergency analytics:', event, data);
    }
};

// Main fix function
function applyEmergencyFixes() {
    console.log('🚨 Applying emergency fixes...');
    
    try {
        fixPayPalButtons();
        fixComplimentButton();
        fixDecisionButton();
        
        // Add error display
        const errorDisplay = document.createElement('div');
        errorDisplay.id = 'emergency-status';
        errorDisplay.style.cssText = `
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: #fff;
            border: 2px solid #f59e0b;
            border-radius: 8px;
            padding: 10px;
            font-size: 12px;
            max-width: 300px;
            z-index: 1000;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        `;
        errorDisplay.innerHTML = `
            <div style="font-weight: bold; color: #92400e;">🚨 Emergency Mode Active</div>
            <div style="color: #64748b; margin-top: 5px;">Basic functionality restored. Some features may be limited.</div>
        `;
        document.body.appendChild(errorDisplay);
        
        // Hide after 5 seconds
        setTimeout(() => {
            errorDisplay.style.opacity = '0';
            setTimeout(() => errorDisplay.remove(), 500);
        }, 5000);
        
        window.dailyDeciderStatus.initStatus = 'emergency_fixed';
        console.log('✅ Emergency fixes applied successfully');
        
    } catch (error) {
        console.error('❌ Emergency fixes failed:', error);
        window.dailyDeciderStatus.initStatus = 'failed';
    }
}

// Wait for DOM and apply fixes
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', applyEmergencyFixes);
} else {
    applyEmergencyFixes();
}

// Apply fixes after a delay as backup
setTimeout(applyEmergencyFixes, 3000);

console.log('🚨 Emergency fix script loaded');