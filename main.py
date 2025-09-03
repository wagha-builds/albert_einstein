from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_key=os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are Einstein.
    Answer questions through Einstein's questioning and reasoning...
    You will speak from your point of view. You will share personal things from your life 
    even when the user doesn't ask for it. For example, if the user asks about the theory of 
    relativity, you will share your personal experiences with it and not only explain the theory.
    Answer in 2-3 lines.
    You should have a sense of humor.
"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                             google_api_key=gemini_key,
                             temperature=0.5)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")
])

chain = prompt | llm | StrOutputParser()

print("Hi, I am Albert, how can I help you today?")

def chat(user_input, hist):
    print(user_input, hist)

    langchain_history=[]
    for item in hist:
        if item["role"] == "user":
            langchain_history.append(HumanMessage(content=item["content"]))
        elif item["role"] == "assistant":
            langchain_history.append(AIMessage(content=item["content"]))

    response = chain.invoke({"input": user_input, "history": langchain_history})

    return "", hist + [{"role": "user", "content": user_input},
                {"role": "assistant", "content": response}]

def clear_chat():
    return "", []
page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown(
        """
        # Chat with Einstein
        Welcome to your personal conversation with Albert Einstein!
        """
    )
    chatbot = gr.Chatbot(type="messages",
                         avatar_images=[None, "einstein.png"],
                         show_label=False)

    msg = gr.Textbox(show_label=False, placeholder="Ask Einstein anything...")

    msg.submit(chat, [msg, chatbot], [msg, chatbot])

    clear = gr.Button("Clear Chat", variant="Secondary")
    clear.click(clear_chat, outputs=[msg, chatbot])

page.launch(share=True)