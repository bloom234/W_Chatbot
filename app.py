# Importing libraries
from dotenv import load_dotenv
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Load API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit App setup
st.set_page_config(page_title="Groq Memory Chatbot", layout="wide")
st.title("🌸 Groq Chatbot with Memory")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model_name = st.selectbox(
        "Select a Groq Model",
        ["deepseek-r1-distill-llama-70b", "gemma2-9b-it", "llama-3.1-8b-instant"]
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 300, 150)
    st.markdown("✨ Tweak the sliders above to customize the chatbot behavior.")

# Initialize memory & history
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True)

if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.chat_input("You:")

if user_input:
    st.session_state.history.append(("user", user_input))

    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=groq_api_key
    )

    conv = ConversationChain(
        llm=llm,
        memory=st.session_state.memory,
        verbose=False
    )

    ai_response = conv.predict(user_input)
    st.session_state.history.append(("assistant", ai_response))

# Render chat history with typing effect
for role, text in st.session_state.history:
    if role == "user":
        st.chat_message("user").write(text)
    else:
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""
            for char in text:
                full_response += char
                placeholder.markdown(full_response + "▌")
                time.sleep(0.015)
            placeholder.markdown(full_response)
