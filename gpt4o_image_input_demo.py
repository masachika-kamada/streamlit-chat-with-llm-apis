import base64
from pathlib import Path

from dotenv import load_dotenv
from langchain.schema import HumanMessage
from langchain_llm_streamer import stream_print
from langchain_openai import AzureChatOpenAI

load_dotenv(Path(__file__).parent / ".env")


llm = AzureChatOpenAI(azure_deployment="gpt-4o", temperature=0)

# PNGファイルのパスを指定
file_path = "imgs/image01.png"

# ファイルをバイナリモードで開いて読み込む
with open(file_path, 'rb') as image_file:
    # ファイルの内容を読み込む
    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

# data URIスキームに従ってフォーマットする
image_url = f"data:image/png;base64,{encoded_string}"

messages = [
    HumanMessage(
        content=[
            { "type": "text", "text": "画像について説明してください" },
            { "type": "image_url", "image_url": { "url": image_url } }
        ]
    )
]

stream_print(llm, messages)
