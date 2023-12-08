import base64
import io
import json
import os
import os.path
import re
import time
from abc import ABC
from typing import Any
from uuid import uuid4

import gradio as gr
import requests
from PIL import Image
from langchain.agents import initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import BaseTool


class AgentBot:
    def __init__(self):
        chat_llm = AzureChatOpenAI(
            azure_endpoint=AZURE_END_POINT,
            openai_api_key=AZURE_OPEN_KEY,
            deployment_name="gpt-35-turbo",
            openai_api_version="2023-10-01-preview",
            temperature=0.0
        )
        # initialize conversational memory
        conversational_memory = ConversationBufferWindowMemory(
            memory_key='chat_history',
            k=5,
            return_messages=True
        )

        # tools = [SdxlImage(api_key=SDXL_API_KEY, api_secret=SDXL_API_SECRET)]

        # initialize agent with tools
        self.agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=[],
            llm=chat_llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=conversational_memory
        )

    def run(self, txt) -> str:
        result = self.agent(txt)
        return result["output"]

    def clear(self):
        self.agent.memory.clear()


bot = AgentBot()

block_css = """#col_container {width: 1000px; margin-left: auto; margin-right: auto;}
                #chatbot {height: 520px; overflow: auto;}"""

with gr.Blocks(css=block_css) as demo:
    gr.Markdown("<h3><center>ChatGPT LangChain</center></h3>")
    gr.Markdown(
        """
         This LangChain GPT can generate SD-XL Image  
        """
    )

    with gr.Row() as input_raw:
        with gr.Column(elem_id="col_container"):
            chatbot = gr.Chatbot([],
                                 elem_id="chatbot",
                                 label="ChatBot LangChain for AIGC",
                                 bubble_full_width=False,
                                 avatar_images=(None, (os.path.join(os.path.dirname(__file__), "avatar.png"))),
                                 )

            msg = gr.Textbox()

    with gr.Row():
        with gr.Column(scale=0.10, min_width=0):
            run = gr.Button("🏃‍♂️Run")
        with gr.Column(scale=0.10, min_width=0):
            clear = gr.Button("🔄Clear️")


    def respond(message, chat_history):
        # bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        bot_message = bot.run(message)
        regx = r'\b[\w-]+\.png'
        match_image = re.findall(regx, bot_message)
        chat_history.append((message, bot_message))
        if match_image:
            for image in match_image:
                image_path = os.path.join(SAVE_FOLDER, image)
                chat_history.append(
                    (None, (image_path,)),
                )
        time.sleep(2)
        return "", chat_history


    def clearMessage():
        # clear agent memory
        bot.clear()

    # execute action
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    run.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clearMessage)
    clear.click(lambda: [], None, chatbot)

    gr.Examples(
        examples=["generate a image about a boy reading books using SDXL",
                  "generate two images about a gril in the classroom using SDXL",
                  ],
        inputs=msg
    )

demo.launch()
