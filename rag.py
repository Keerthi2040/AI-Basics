## RAG Q&A Conversation With PDF Including Chat History
#Streamlit (st): Used to create the web interface for the application.
import streamlit as st
# create_retrival_chain is used to create a retrival chain
# create_history_aware_retriver will add the history. used for retrival with chat history functionality
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
# create_stuff_documents_chain will combine all the documents and send it to context
from langchain.chains.combine_documents import create_stuff_documents_chain
# Vector Store DB
#Chroma (Chroma): A vector store used to store and retrieve document embeddings.
#ChatMessageHistory: Manages the chat history.
#BaseChatMessageHistory: Base class for chat history management
#ChatPromptTemplate: Used to create prompts for the LLM.
#MessagesPlaceholder: A placeholder for chat history in the prompt.
#ChatGroq: Interface to interact with the Groq API for LLM.
#RunnableWithMessageHistory: Adds chat history to the runnable chain.
#HuggingFaceEmbeddings: Generates embeddings for the documents using Hugging Face models.
#RecursiveCharacterTextSplitter: Splits text into chunks for processing.
#PyPDFLoader: Loads PDF documents.
#os: Used to interact with the operating system, e.g., environment variables.
#dotenv: Loads environment variables from a .env file.
from langchain_chroma import Chroma 
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

from dotenv import load_dotenv
load_dotenv()

#langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv("****","default_langchain_api_key")
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Chat_with_PDF"

#os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
#Initializes the embedding model (all-MiniLM-L6-v2) from Hugging Face, which is used to convert text into vectors for retrieval.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#setup streamlit
st.title("Conversational RAG with PDF upload and Chat history")
st.write("This application enables us to upload multiple pdf files and chat with their contents.") 
st.write("This application is powered by Gemma2-9b-IT model from Groq and all-MiniLM-L6-v2 Embedding model from Huggingface")
st.write("The entire application is tracked using LangSmith")

# Input the Groq API key
api_key = st.text_input("Enter the Groq API key:",type="password")

# Check if Groq API key is provided
if api_key:
    #api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-IT")
    
    # Chat interface
    session_id=st.text_input("Session ID", value="default_session")
    # Statefully manage chat history
    
    if 'store' not in st.session_state:
        st.session_state.store={}
    
    uploaded_files = st.file_uploader("Choose a PDF file",type="pdf", accept_multiple_files=True)
    
    # Process uploaded pdf
    if uploaded_files:
        documents=[]
        for uploaded_file in uploaded_files:
            tempdf=f"./temp.pdf"
            with open(tempdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name
    
            loader=PyPDFLoader(tempdf)
            docs=loader.load()
            documents.extend(docs)
        
        # Split and create embeddings for the documents
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits=text_splitter.split_documents(documents)
        vector_store=Chroma.from_documents(documents=splits,embedding=embeddings)
        retriever=vector_store.as_retriever()
    
        contextualize_q_system_prompt=(
            "Given a chat history and the latest user question"
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
    
        history_aware_retriver=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)
    
        # Answer question
        system_prompt=(
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )
    
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )
    
        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retriver,question_answer_chain)
    
        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()
            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
            rag_chain,get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )
    
        user_input=st.text_input("Your_question:")
        if user_input:
            session_history=get_session_history(session_id)
            response=conversational_rag_chain.invoke(
                {"input":user_input},
                config={
                    "configurable": {"session_id":session_id}
                }
            )
            st.write(st.session_state.store)
            st.write("Assistant:",response['answer'])
            st.write("Chat History",session_history.messages)

else:
    st.warning("Please enter the Groq API key")





