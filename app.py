# Importing librarires
from dotenv import load_dotenv
import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# load api key
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# Streamlit App setup 
st.set_page_config(page_title="Groq Memory Chatbot")
st.title("🌸 Groq Chatbot with Memory")

# Sidebar controls with 

#st.sidebar: st.header("⚙️ Settings")

model_name = st.sidebar.selectbox(
    "Please Select any Groq Model",
    ["deepseek-r1-distill-llama-70b","gemma2-9b-it","llama-3.1-8b-instant"]
)
temperature = st.sidebar.slider(
    "Temperature/Tuning" , 0.0, 1.0,0.7
)
max_tokens = st.sidebar.slider(
    "Max Tokens", 50,300, 150
)

## Initialize memory & history
if "memory" not in st.session_state:
    # persist memory across reruns
    st.session_state.memory = ConversationBufferMemory(
        return_messages=True 
    )

if "history" not in st.session_state:
    st.session_state.history =[]


## user input
user_input = st.chat_input("You: ")

if user_input:
    # append user turn to visible history
    st.session_state.history.append(("user",user_input))

    # instantiate a fressh llm for this turn
    llm = ChatGroq(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # build conversationchain with our memory
    conv = ConversationChain(
        llm=llm,
        memory = st.session_state.memory,
        verbose=False
    )
    
    # get AI response (memory is updated internally)
    ai_response = conv.predict(input=user_input)

    ## append assistant turn to visible history
    st.session_state.history.append(("assistant",ai_response))

# render chat bubble
for role, text in st.session_state.history:
    if role == 'user':
        st.chat_message("user").write(text) # user style
    else:
        st.chat_message("assistant").write(text) # assistant style

# Render chat history with typing effect
import time

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

# --- Custom Theme Styling ---
st.markdown("""
    <style>
    /* Main background */
    .stApp {
        background-color: #ffd6e8; /* soft pink */
    }

    /* Sidebar background */
    [data-testid="stSidebar"] {
        background-color: #ffb6c1; /* light pink */
    }

    /* Titles and headers */
    .stMarkdown, .stHeader, .stText, h1, h2, h3 {
        color: black !important;
    }

    /* Dropdown and sliders */
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: black !important;
        color: white !important;
    }

    .stSlider > div > div > div[data-baseweb="slider"] {
        background-color: black !important;
    }

    /* Chat input box */
    div[data-testid="stChatInput"] textarea {
        background-color: black !important;
        color: white !important;
        border: 2px solid #ff69b4; /* hot pink border */
        border-radius: 10px;
    }

    /* User messages */
    div[data-testid="stChatMessageUser"] {
        background-color: #ff69b4 !important; /* pink bubble */
        color: white !important;
        border-radius: 12px;
    }

    /* Assistant messages */
    div[data-testid="stChatMessageAssistant"] {
        background-color: white !important;
        color: black !important;
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)
