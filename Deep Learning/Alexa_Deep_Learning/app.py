import json
from flask import Flask, render_template, Response, stream_with_context
import alexa_core

app = Flask(__name__, template_folder='templates', static_folder='frontend')

# Start by loading the models
print("Initializing Alexa Core...")
alexa_core.init_alexa()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/interact')
def interact():
    # We use Server-Sent Events (SSE) to update the UI on progress
    def generate():
        # 1. Listening
        yield f"data: {json.dumps({'state': 'listening'})}\n\n"
        audio_path = alexa_core.record_audio()
        
        # 2. Transcribing
        yield f"data: {json.dumps({'state': 'transcribing'})}\n\n"
        text = alexa_core.speech_to_text(audio_path)
        
        # 3. Processing parameters and taking action
        yield f"data: {json.dumps({'state': 'processing', 'text': text})}\n\n"
        intent = alexa_core.predict_intent(text)
        action_response = alexa_core.perform_action(intent, text)
        
        # 4. Speaking response
        yield f"data: {json.dumps({'state': 'speaking', 'response': action_response})}\n\n"
        alexa_core.speak(action_response)
        
        # 5. Idle
        yield f"data: {json.dumps({'state': 'idle'})}\n\n"
        
    return Response(stream_with_context(generate()), mimetype="text/event-stream")

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000, threaded=True)
