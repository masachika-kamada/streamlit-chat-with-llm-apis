import base64
import os
from io import BytesIO
from pathlib import Path

import streamlit as st
import streamlit_authenticator as stauth
from dotenv import load_dotenv
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from st_img_pastebutton import paste

load_dotenv(Path(__file__).parent / ".env")


class LLMChatManager:
    def __init__(self):
        self.system_prompt = "あなたは愉快なAIです。ユーザの入力に日本語で答えてください"
        self.model = "gpt-4o"
        self.mode = {
            "Chatbot Responses": {"temperature": 0.5, "top_p": 0.5},
            "Creative Writing": {"temperature": 0.7, "top_p": 0.8},
            "Code Generation": {"temperature": 0.2, "top_p": 0.1},
            "Code Comment Generation": {"temperature": 0.3, "top_p": 0.2},
            "Exploratory Code Writing": {"temperature": 0.6, "top_p": 0.7}
        }
        self.selected_mode = list(self.mode.keys())[0]
        self.temperature = self.mode[self.selected_mode]["temperature"]
        self.top_p = self.mode[self.selected_mode]["top_p"]
        self.llm = AzureChatOpenAI(model=self.model, temperature=self.temperature, top_p=self.top_p)
        self.image_url = None
        if "messages" not in st.session_state:
            self.init_messages()
        if "rerender_chat" not in st.session_state:
            st.session_state.rerender_chat = False

    def select(self):
        with st.sidebar:
            st.sidebar.title("🧠 LLM Chat")
            self._prompt_options()
            self._image_input_options()
            self.clear_conversation_button()

    def _prompt_options(self):
        with st.expander("Settings", expanded=True):
            self.system_prompt = st.text_area("System Prompt", self.system_prompt, height=120, on_change=self._rerender)
            self.n_history = st.slider("Number of History", 1, 14, 10, 1, help="Number of previous messages to consider", on_change=self._rerender)
            self.selected_mode = st.radio("Mode", list(self.mode.keys()), on_change=self._update_and_rerender)

    def _image_input_options(self):
        with st.expander("Image Input", expanded=False):
            self.use_image = st.toggle("Use Image Input", False, on_change=self._rerender)

            image_data = paste(label="paste from clipboard")
            if image_data is not None:
                header, encoded = image_data.split(",", 1)
                self.image_url = f"data:image/png;base64,{encoded}"
                binary_data = base64.b64decode(encoded)
                bytes_data = BytesIO(binary_data)
                st.image(bytes_data, use_column_width=True)
                self._rerender()

    def _update_and_rerender(self):
        self._update_llm()
        self._rerender()

    def _update_llm(self):
        self.temperature = self.mode[self.selected_mode]["temperature"]
        self.top_p = self.mode[self.selected_mode]["top_p"]
        self.llm = AzureChatOpenAI(model=self.model, temperature=self.temperature, top_p=self.top_p)

    def _rerender(self):
        st.session_state.rerender_chat = True

    def clear_conversation_button(self):
        st.sidebar.button("Clear Conversation", on_click=self.init_messages)

    def init_messages(self):
        st.session_state.messages = [SystemMessage(content=self.system_prompt)]


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
    st.markdown(
        """
        <style>
        body, div {
            font-family: 'Source Sans Pro', sans-serif !important;
            font-size: 16px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    llm_chat_manager = LLMChatManager()
    llm_chat_manager.select()

    if st.session_state.rerender_chat:
        display_chat_history()
        st.session_state.rerender_chat = False

    user_input = st.chat_input("Message...")

    if user_input:
        if llm_chat_manager.use_image and llm_chat_manager.image_url is not None:
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

        response = display_stream(llm_chat_manager.llm.stream(input_messages))
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
