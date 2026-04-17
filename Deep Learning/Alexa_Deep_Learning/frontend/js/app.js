const listenBtn = document.getElementById('listen-btn');
const orb = document.getElementById('orb');
const statusText = document.getElementById('status-text');
const userTranscript = document.getElementById('user-transcript');
const transcriptText = document.getElementById('transcript-text');
const botResponse = document.getElementById('bot-response');
const responseText = document.getElementById('response-text');

let isInteractionInProgress = false;

listenBtn.addEventListener('click', () => {
    if (isInteractionInProgress) return;
    
    startInteraction();
});

function startInteraction() {
    isInteractionInProgress = true;
    listenBtn.disabled = true;
    
    // Reset UI
    userTranscript.style.display = 'none';
    botResponse.style.display = 'none';
    
    // Connect to SSE Endpoint
    const eventSource = new EventSource('/api/interact');
    
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleStateChange(data);
            
            if (data.state === 'idle') {
                eventSource.close();
                isInteractionInProgress = false;
                listenBtn.disabled = false;
            }
        } catch (e) {
            console.error("Error parsing event data:", e);
        }
    };
    
    eventSource.onerror = (error) => {
        console.error("SSE Error:", error);
        eventSource.close();
        isInteractionInProgress = false;
        listenBtn.disabled = false;
        setOrbState('idle', 'Error communicating with Alexa.');
    };
}

function handleStateChange(data) {
    const state = data.state;
    
    switch (state) {
        case 'listening':
            setOrbState('listening', 'Alexa is listening...');
            break;
            
        case 'transcribing':
            setOrbState('transcribing', 'Transcribing audio...');
            break;
            
        case 'processing':
            setOrbState('processing', 'Thinking...');
            // Show user text
            userTranscript.style.display = 'block';
            transcriptText.textContent = data.text || "...";
            break;
            
        case 'speaking':
            setOrbState('speaking', 'Alexa is responding...');
            // Show bot response
            botResponse.style.display = 'block';
            const responseString = data.response || "Done.";
            responseText.textContent = responseString;
            
            // Speak using the browser's high-quality Neural Web Speech API
            const msg = new SpeechSynthesisUtterance(responseString);
            const voices = window.speechSynthesis.getVoices();
            // Attempt to find a premium voice
            const premiumVoice = voices.find(v => v.name.includes("Google UK English Female") || v.name.includes("Samantha") || v.name.includes("Fiona"));
            if (premiumVoice) msg.voice = premiumVoice;
            
            window.speechSynthesis.speak(msg);
            break;
            
        case 'idle':
            setOrbState('idle', 'Ready to help.');
            break;
            
        default:
            console.warn("Unknown state:", state);
            break;
    }
}

function setOrbState(stateClass, text) {
    // Remove all state classes
    orb.className = 'orb';
    
    // Add new state class
    orb.classList.add(`state-${stateClass}`);
    
    // Update text
    statusText.textContent = text;
}
