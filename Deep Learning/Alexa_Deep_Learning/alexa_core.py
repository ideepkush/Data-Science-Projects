import os
import sys
import webbrowser
import datetime
import random
import re
import pandas as pd
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

# Globals to hold models
model = None
vectorizer = None
intent_model = None
shopping_list = []

def init_alexa():
    global model, vectorizer, intent_model
    
    # 2. Whisper Model
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # 3. Intent Model (scikit-learn)
    print("Training intent model...")
    alexa_df = pd.read_csv("alexa_data.csv")
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(alexa_df["prompt"])
    intent_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=500, random_state=42)
    intent_model.fit(X, alexa_df["intent"])
    print("Alexa initialization complete.")

def record_audio(filename="input.wav", duration=5, fs=48000):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, recording)
    return filename

def speech_to_text(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

def predict_intent(text):
    if not text.strip():
        return "none"
    X_test = vectorizer.transform([text])
    intent = intent_model.predict(X_test[0])
    return intent[0]

def speak(text):
    # Text-to-speech is now handled natively by the web browser frontend 
    # for much more fluid and natural voice quality!
    print(f"Server generated response: {text}")
    pass

def perform_action(intent, prompt=""):
    # 🎵 PLAY MUSIC
    if intent == "play_music":
        webbrowser.open("https://www.youtube.com/results?search_query=music")
        return "Playing music on YouTube"

    # 🌐 OPEN WEBSITE
    elif intent == "open_website":
        prompt_lower = prompt.lower()
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', prompt_lower)

        if "youtube" in cleaned or "you tube" in cleaned:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"
        elif "google" in prompt_lower:
            webbrowser.open("https://www.google.com")
            return "Opening Google"
        elif "github" in prompt_lower:
            webbrowser.open("https://www.github.com")
            return "Opening GitHub"
        else:
            return "Which website would you like me to open?"

    # 😂 JOKES / FUN
    elif intent == "jokes_fun":
        jokes = [
            "Why did the computer get cold? Because it forgot to close Windows!",
            "Why do programmers prefer dark mode? Because light attracts bugs!",
            "I told my AI assistant a joke… it said, processing humor module."
        ]
        return random.choice(jokes)

    # 📰 NEWS
    elif intent == "news":
        webbrowser.open("https://news.google.com")
        return "Here are today’s top headlines"

    # 🎬 MOVIES
    elif intent == "movies":
        webbrowser.open("https://www.imdb.com/chart/top/")
        return "Here are some popular movies you can watch"

    # ⏰ TIMER
    elif intent == "set_timer":
        minutes = re.findall(r'\d+', prompt)
        minutes = int(minutes[0]) if minutes else 1
        return f"Timer set for {minutes} minutes"

    # ⏰ ALARM
    elif intent == "alarm":
        time_match = re.search(r'(\d{1,2})(?::(\d{2}))?\s*(am|pm)?', prompt.lower())
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            period = time_match.group(3) if time_match.group(3) else "am"
            return f"Alarm set for {hour}:{str(minute).zfill(2)} {period.upper()}"
        return "Alarm set"

    # 🔔 REMINDER
    elif intent == "reminder":
        return "Reminder saved"

    # 🕒 DATE & TIME
    elif intent == "date_time":
        now = datetime.datetime.now()
        return f"It is {now.strftime('%I:%M %p')} on {now.strftime('%B %d, %Y')}"

    # 📅 CALENDAR
    elif intent == "calendar":
        return "You have no upcoming events today"

    # 🌦 WEATHER
    elif intent == "weather":
        prompt_lower = prompt.lower()
        match = re.search(r"(in|of)\s+([a-zA-Z\s]+)", prompt_lower)

        if match:
            city = match.group(2).strip()
            webbrowser.open(f"https://www.google.com/search?q=weather+in+{city.replace(' ', '+')}")
            return f"Showing current weather in {city.title()}"
        else:
            webbrowser.open("https://www.google.com/search?q=weather+today")
            return "Here is the latest weather forecast"

    # 🧠 GENERAL Q&A
    elif intent == "general_qa":
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["temperature", "weather", "forecast"]):
            match = re.search(r"(in|of)\s+([a-zA-Z\s]+)", prompt_lower)
            if match:
                city = match.group(2).strip()
                webbrowser.open(f"https://www.google.com/search?q=weather+in+{city.replace(' ', '+')}")
                return f"Showing current weather in {city.title()}"

        query = prompt.replace(" ", "+")
        webbrowser.open(f"https://www.google.com/search?q={query}")
        return "Here is what I found on the web"

    # 🌍 FACTS
    elif intent == "facts":
        facts = [
            "Honey never spoils.",
            "Bananas are berries, but strawberries are not.",
            "Octopuses have three hearts."
        ]
        return random.choice(facts)

    # 🚗 TRAFFIC
    elif intent == "traffic":
        webbrowser.open("https://www.google.com/maps/dir/Home/Office")
        return "Checking traffic on your usual route"

    # 🧭 DIRECTIONS
    elif intent == "directions":
        location = prompt.lower().replace("directions to", "").strip()
        webbrowser.open(f"https://www.google.com/maps/search/{location}")
        return f"Showing directions to {location}"

    # 🛒 SHOPPING LIST
    elif intent == "shopping_list":
        item = prompt.lower().replace("add", "").replace("to my shopping list", "").strip()
        shopping_list.append(item)
        return f"Added {item} to your shopping list"

    # 💡 SMART HOME
    elif intent == "smart_home":
        return "Smart home command executed"

    # 🤖 PERSONALITY
    elif intent == "personality":
        responses = [
            "I’m doing great! Ready to help you.",
            "I am your friendly AI assistant!",
            "I was created to make your life easier."
        ]
        return random.choice(responses)

    # 🧮 CALCULATOR
    elif intent == "calculator":
        nums = list(map(int, re.findall(r'\d+', prompt)))
        if "plus" in prompt and len(nums) >= 2:
            return f"The answer is {nums[0] + nums[1]}"
        elif "minus" in prompt and len(nums) >= 2:
            return f"The answer is {nums[0] - nums[1]}"
        elif "times" in prompt and len(nums) >= 2:
            return f"The answer is {nums[0] * nums[1]}"
        elif "divide" in prompt and len(nums) >= 2 and nums[1] != 0:
            return f"The answer is {nums[0] / nums[1]}"
        else:
            return "Sorry, I couldn’t calculate that."

    # ❓ UNKNOWN
    else:
        return "Sorry, I didn’t understand that command."
