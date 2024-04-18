from langchain import hub
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
import getpass
import os

####### get your api keys ###############

os.environ["OPENAI_API_KEY"] = getpass.getpass()

prompt = hub.pull("rlm/rag-prompt")

############# Indexing: Load #############################
loader = TextLoader("./sample.txt")
docs = loader.load()

###############  Indexing: Split #########################

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

###############  Indexing: Store #########################

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

################ Retriever ###########################
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

###########################################################
################ USING GPT3.5-TURBO #######################
###########################################################
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


###########################################################
################ history_aware_retriever###################
###########################################################

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


###########################################################
################ question_answer_chain#####################
###########################################################

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

###########################################################
################ rag_chain#####################
###########################################################

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

if __name__ == "__main__":
    chat_history = []
    question = "Who won men's cricket World Cup 2023? ?"
    ai_msg = rag_chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), ai_msg["answer"]])
    print(ai_msg["answer"])
