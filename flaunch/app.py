import streamlit as st
from llama_cpp import Llama
from textblob import TextBlob
from deep_translator import GoogleTranslator
import asyncio
import functools
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import Dict, List, Any

# Define all the text content in English
TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "English": {
        "page_title": "Mental Health Chatbot",
        "settings": "Settings",
        "select_language": "Select Response Language",
        "mood_tracking": "Mood Tracking",
        "personalization": "Personalization",
        "about_chatbot": "About this Chatbot",
        "chatbot_description": "This AI-powered chatbot provides proactive mental health support in multiple languages.",
        "disclaimer": "Please note: This is not a substitute for professional mental health care.",
        "main_title": "🤖 Mental Health Support Chatbot",
        "input_placeholder": "Type your message here...",
        "thinking": "Thinking...",
        "recommended_resources": "Recommended Resources:",
        "footer": "Remember, this chatbot is not a substitute for professional mental health care. If you're experiencing severe distress, please seek help from a qualified mental health professional.",
        "your_name": "Your Name",
        "your_age": "Your Age",
        "your_interests": "Your Interests",
        "preferred_topics": "Preferred Conversation Topics",
        "how_feeling": "How are you feeling today?",
        "log_mood": "Log Mood",
        "mood_logged": "Mood logged successfully!",
        "mood_over_time": "Your Mood Over Time",
        "initial_question": "Hello! I'm here to support your mental well-being. How are you feeling today?",
        "crisis_response": "I'm concerned about what you've shared. If you're having thoughts of harming yourself, please reach out to a crisis helpline or a mental health professional immediately. You're not alone, and help is available."
    }
}

@st.cache_resource
def load_model() -> Llama:
    try:
        with st.spinner("Loading model. This may take a moment..."):
            model_path = "meta-llama-3.1-8b-instruct.f16.gguf"
            model = Llama(model_path=model_path, n_ctx=1024, n_batch=512)
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.markdown(f" ")
        return None

@st.cache_data
def analyze_sentiment(text: str) -> float:
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

@st.cache_data
def generate_text(_model: Llama, prompt: str, max_tokens: int = 150) -> str:
    try:
        output = _model(prompt, max_tokens=max_tokens, stop=["Human:", "\n"], echo=False)
        return output['choices'][0]['text'].strip()
    except Exception as e:
        st.error(f"Error generating text: {str(e)}")
        return "I'm sorry, I encountered an error while processing your request."

@st.cache_data
def translate_text(text: str, target_language: str) -> str:
    try:
        translator = GoogleTranslator(source='auto', target=target_language)
        return translator.translate(text)
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

def translate_content(content: Dict[str, str], target_language: str) -> Dict[str, str]:
    if target_language == "English":
        return content
    translator = GoogleTranslator(source='en', target=target_language)
    return {k: translator.translate(v) for k, v in content.items()}

def get_translated_content(language: str) -> Dict[str, str]:
    if language not in TRANSLATIONS:
        content = TRANSLATIONS["English"]
        return translate_content(content, language)
    return TRANSLATIONS[language]

async def run_parallel(*functions):
    return await asyncio.gather(*[asyncio.to_thread(func) for func in functions])

def track_mood(t: Dict[str, str]):
    if 'mood_history' not in st.session_state:
        st.session_state.mood_history = []

    mood = st.multiselect(t["how_feeling"], ["Sad", "Happy", "Angry", "Anxious", "Feared"])
    if st.button(t["log_mood"]):
        st.session_state.mood_history.append({
            'date': datetime.now().strftime("%Y-%m-%d"),
            'mood': mood
        })
        st.success(t["mood_logged"])

    if st.session_state.mood_history:
        df = pd.DataFrame(st.session_state.mood_history)
        fig = px.line(df, x='date', y='mood', title=t["mood_over_time"])
        st.plotly_chart(fig)

def personalize_chatbot(t: Dict[str, str]) -> Dict[str, Any]:
    if 'user_preferences' not in st.session_state:
        st.session_state.user_preferences = {
            'name': '',
            'age': '',
            'interests': [],
            'preferred_topics': []
        }

    st.session_state.user_preferences['name'] = st.text_input(t["your_name"], st.session_state.user_preferences['name'])
    st.session_state.user_preferences['age'] = st.number_input(t["your_age"], min_value=0, max_value=120, value=st.session_state.user_preferences['age'] if st.session_state.user_preferences['age'] else 0)
    interests = st.multiselect(t["your_interests"], ["Reading", "Music", "Sports", "Art", "Travel", "Technology"], default=st.session_state.user_preferences['interests'])
    st.session_state.user_preferences['interests'] = interests
    topics = st.multiselect(t["preferred_topics"], ["Stress Management", "Positive Thinking", "Mindfulness", "Relationship Advice", "Career Guidance"], default=st.session_state.user_preferences['preferred_topics'])
    st.session_state.user_preferences['preferred_topics'] = topics

    return st.session_state.user_preferences

def recommend_resources(sentiment: float) -> List[str]:
    resources = {
        "positive": [
            "Mindfulness meditation app",
            "Gratitude journaling guide",
            "Positive affirmations list"
        ],
        "neutral": [
            "Self-care checklist",
            "Stress management techniques",
            "Healthy habit tracker"
        ],
        "negative": [
            "Crisis helpline numbers",
            "Therapy finder tool",
            "Coping strategies for difficult emotions"
        ]
    }

    if sentiment > 0.2:
        category = "positive"
    elif sentiment < -0.2:
        category = "negative"
    else:
        category = "neutral"

    return resources[category]

def detect_crisis(text: str) -> bool:
    crisis_keywords = ["suicide", "kill myself", "want to die", "end it all"]
    return any(keyword in text.lower() for keyword in crisis_keywords)

def run_app():
    st.set_page_config(page_title="Mental Health Chatbot", page_icon="🤖", layout="wide")

    st.markdown("""
        <style>
        .stApp {
            background-color: #1E1E1E;
            color: #FFFFFF;
        }
        .stTextInput > div > div > input {
            background-color: #2B2B2B;
            color: #FFFFFF;
        }
        .stSelectbox > div > div > select {
            background-color: #2B2B2B;
            color: #FFFFFF;
        }
        .chat-container {
            background-color: #2B2B2B;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    model = load_model()
    

    with st.sidebar:
        st.title("Settings")
        language = st.selectbox("Select Response Language", ["English", "Hindi", "Gujarati", "Marathi"])
        language_code = {"English": "en", "Hindi": "hi", "Gujarati": "gu", "Marathi": "mr"}[language]

        if 'language' not in st.session_state or st.session_state.language != language:
            st.session_state.language = language
            st.session_state.translations = get_translated_content(language_code)
            st.rerun()

        t = st.session_state.translations

        st.divider()
        st.subheader(t["personalization"])
        user_preferences = personalize_chatbot(t)
        st.divider()
        st.subheader(t["mood_tracking"])
        track_mood(t)
        st.divider()
        st.write(t["about_chatbot"])
        st.write(t["chatbot_description"])
        st.write(t["disclaimer"])

    st.title(t["main_title"])

    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False

    if not st.session_state.conversation_started:
        initial_question = t["initial_question"]
        st.session_state.messages.append({"role": "assistant", "content": initial_question})
        st.session_state.conversation_started = True

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(t["input_placeholder"])

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        user_input_en, sentiment_score = loop.run_until_complete(run_parallel(
            functools.partial(translate_text, user_input, 'en') if language != "English" else lambda: user_input,
            functools.partial(analyze_sentiment, user_input)
        ))

        sentiment = "positive" if sentiment_score > 0 else "neutral" if sentiment_score == 0 else "negative"

        if detect_crisis(user_input_en):
            crisis_response = t["crisis_response"]
            st.session_state.messages.append({"role": "assistant", "content": crisis_response})
            with st.chat_message("assistant"):
                st.markdown(crisis_response)
            st.stop()

        formatted_history = "\n".join(
            f"{'Assistant' if msg['role'] == 'assistant' else 'Human'}: {msg['content']}"
            for msg in st.session_state.messages[-5:]
        )

        response_prompt = f"""
        You are a compassionate AI assistant dedicated to supporting mental well-being. Based on the conversation history and the user's current sentiment, craft a thoughtful response that addresses the user's input and provides support. Then, ask a follow-up question to encourage further discussion.

        User Information:
        Name: {user_preferences['name']}
        Age: {user_preferences['age']}
        Interests: {', '.join(user_preferences['interests'])}
        Preferred Topics: {', '.join(user_preferences['preferred_topics'])}

        Conversation History:
        {formatted_history}

        User's Input:
        Human: {user_input_en}

        Detected Sentiment: {sentiment}

        Your Response with emoji and Follow-up Question:
        Assistant:
        """

        with st.spinner(t["thinking"]):
            response_en = generate_text(model, response_prompt)
            response = translate_text(response_en, language_code) if language != "English" else response_en
            st.session_state.messages.append({"role": "assistant", "content": response})

            with st.chat_message("assistant"):
                st.markdown(response)

            resources = recommend_resources(sentiment_score)
            st.subheader(t["recommended_resources"])
            for resource in resources:
                st.write(f"- {resource}")

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown(t["footer"])

if __name__ == "__main__":
    run_app()
