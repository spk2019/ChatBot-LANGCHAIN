import gradio as gr
from langchain_core.messages import HumanMessage
from add_chat_history import rag_chain


chat_history = []
import gradio as gr

def ask(question, history):
    ai_message = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_message["answer"]])
    return ai_message['answer']

demo = gr.ChatInterface(fn=ask, examples=["Hello, my name is shakti","which team won 2023 cricket world cup?"], title="Cricket Chat Bot",theme=gr.themes.Soft())



if __name__ == "__main__":
    gr.close_all()
    demo.launch(share = True)
