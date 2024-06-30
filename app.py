import base64
import os
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_cohere import ChatCohere
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import AzureChatOpenAI
from st_img_pastebutton import paste

load_dotenv(Path(__file__).parent / ".env")


class LLMChatManager:
    def __init__(self):
        self.providers = ["OpenAI", "Google", "Cohere", "Groq"]
        self.models = {
            # デプロイ作成時、デプロイ名とモデル名が同じになるように設定
            "OpenAI": ["gpt-4o", "gpt-35-turbo-instruct"],
            "Google": ["gemini-1.5-flash", "gemini-1.5-pro"],
            "Cohere": ["command-r-plus", "command-r", "command", \
                       "command-light", "command-nightly", "command-light-nightly"],
            "Groq": ["llama3-70b-8192", "llama3-8b-8192"],
        }
        self.system_prompt = "あなたは愉快なAIです。ユーザの入力に日本語で答えてください"
        self.provider = self.providers[0]
        self.model = self.models[self.provider][0]
        self.temperature = 0
        self.image_url = None
        if "messages" not in st.session_state:
            self.init_messages()

    def select(self):
        with st.sidebar:
            st.sidebar.title("🧠 LLM Chat")
            self._model_options()
            self._prompt_options()
            self._image_input_options()
            self.clear_conversation_button()

    def _model_options(self):
        with st.expander(f"Model Options"):
            self.provider = st.radio("Select Provider", self.providers, on_change=self.init_messages)
            self.model = st.selectbox("Select Model", self.models[self.provider], on_change=self.init_messages)
            self.temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.01)

    def _prompt_options(self):
        with st.expander("Prompt Options"):
            self.system_prompt = st.text_area("System Prompt", self.system_prompt, height=140)
            self.n_history = st.slider("Number of History", 1, 10, 8, 1, help="Number of previous messages to consider")

    def _image_input_options(self):
        with st.expander("Image Input Options", expanded=True):
            st.warning("Only available with GPT-4o. Changing options temporarily hides chat, but it reappears after sending a message.", icon="⚠️")
            self.use_image = st.toggle("Use Image Input", False)

            image_data = paste(label="paste from clipboard")
            if image_data is not None:
                header, encoded = image_data.split(",", 1)
                self.image_url = f"data:image/png;base64,{encoded}"
                binary_data = base64.b64decode(encoded)
                bytes_data = BytesIO(binary_data)
                st.image(bytes_data, use_column_width=True)

    def clear_conversation_button(self):
        st.sidebar.button("Clear Conversation", on_click=self.init_messages)

    def init_messages(self):
        st.session_state.messages = [SystemMessage(content=self.system_prompt)]

    def get_llm_instance(self):
        llm_classes = {
            "OpenAI": AzureChatOpenAI,
            "Google": ChatGoogleGenerativeAI,
            "Groq": ChatGroq,
            "Cohere": ChatCohere
        }
        return llm_classes[self.provider](model=self.model, temperature=self.temperature)


def display_chat_history():
    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.text(message.content)


def display_stream(generater):
    with st.chat_message("assistant"):
        return st.write_stream(generater)


def main():
    llm_chat_manager = LLMChatManager()
    llm_chat_manager.select()

    user_input = st.chat_input("Message...")

    if user_input:
        llm = llm_chat_manager.get_llm_instance()

        if llm_chat_manager.model == "gpt-4o" \
                and llm_chat_manager.use_image \
                and llm_chat_manager.image_url is not None:
            st.session_state.messages.append(HumanMessage(
                content=[
                    {"type": "text", "text": user_input},
                    {"type": "image_url", "image_url": {"url": llm_chat_manager.image_url}}
                ]
            ))
            llm_chat_manager.image_url = None
        else:
            st.session_state.messages.append(HumanMessage(content=user_input))
        display_chat_history()

        if len(st.session_state.messages) <= llm_chat_manager.n_history + 1:
            input_messages = st.session_state.messages
        else:
            input_messages = [st.session_state.messages[0]] + st.session_state.messages[-llm_chat_manager.n_history:]

        response = display_stream(llm.stream(input_messages))
        st.session_state.messages.append(AIMessage(content=response))


if __name__ == "__main__":
    st.set_page_config(
        page_title="LLM Chat",
        page_icon="🧠",
        layout="wide",
    )

    authenticator = stauth.Authenticate(
        {"usernames": {
            os.getenv("LOGIN_USERNAME"): {
                "name": os.getenv("LOGIN_NAME"),
                "password": os.getenv("LOGIN_PASSWORD")}}},
        "", "", 30,
    )
    authenticator.login()

    if st.session_state["authentication_status"]:
        main()
    elif st.session_state["authentication_status"] is False:
        st.error("Username/password is incorrect")
    elif st.session_state["authentication_status"] is None:
        st.warning("Please enter your username and password")
