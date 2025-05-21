import pandas as pd
import torch
import streamlit as st
import os
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class RAGChatbot:
    def __init__(self, data_path):
        self.data_path = data_path
        self.vectorstore_path = "vectorstore.pkl"
        self.data = self.load_data(data_path)
        
        if os.path.exists(self.vectorstore_path):
            st.info("Loading cached vectorstore...")
            self.vectorstore = self.load_vectorstore()
        else:
            st.info("Building vectorstore for first time...")
            self.embeddings = self.initialize_embeddings()
            self.vectorstore = self.build_vectorstore()
            self.save_vectorstore()
            
        self.llm = self.initialize_llm()
        self.qa_chain = self.setup_qa_chain()
    
    def load_data(self, data_path):
        df = pd.read_excel(data_path)
        return df
    
    def initialize_embeddings(self):
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    def initialize_llm(self):
        model_id = "facebook/opt-125m"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
        
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    
    def build_vectorstore(self):
        contexts = self.data['context'].tolist()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        documents = [{"content": context, "metadata": {}} for context in contexts]
        chunks = text_splitter.create_documents([doc["content"] for doc in documents])
        
        embeddings = self.initialize_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    
    def save_vectorstore(self):
        with open(self.vectorstore_path, "wb") as f:
            pickle.dump(self.vectorstore, f)
    
    def load_vectorstore(self):
        with open(self.vectorstore_path, "rb") as f:
            return pickle.load(f)
    
    def setup_qa_chain(self):
        template = """
        Answer the question based on the retrieved context. If the answer isn't in the context, say you don't have enough information.
        
        Context: {context}
        Question: {question}
        
        Provide a direct answer without repeating the question or showing the context.
        """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=False
        )
        
        return qa_chain
    
    def ask(self, question):
        answer = self.qa_chain.invoke({"query": question})
        return answer["result"]

def main():
    st.set_page_config(
        page_title="RAG Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Custom CSS for improved design
    st.markdown("""
    <style>
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
        color: #ffffff;
        background-color: #1a1a2e;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        padding: 20px;
    }
    .chat-message {
        padding: 15px 20px;
        border-radius: 12px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .chat-message:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .chat-message.user {
        background-color: #4a69bb;
        margin-left: 20%;
        border-top-right-radius: 0;
    }
    .chat-message.assistant {
        background-color: #2c3e50;
        margin-right: 20%;
        border-top-left-radius: 0;
    }
    .chat-message .message {
        color: #ffffff;
        font-size: 16px;
        line-height: 1.5;
    }
    .stTextInput > div > div > input {
        background-color: #2c3e50;
        color: #ffffff;
        border-radius: 12px;
        padding: 12px 20px;
        border: 1px solid #4a69bb;
        transition: border-color 0.2s ease;
    }
    .stTextInput > div > div > input:focus {
        border-color: #6b7280;
        outline: none;
        box-shadow: 0 0 0 3px rgba(74, 105, 187, 0.3);
    }
    h1 {
        font-size: 2.5em;
        color: #ffffff;
        text-align: center;
        margin-bottom: 10px;
    }
    h4 {
        font-size: 1.2em;
        color: #a0aec0;
        text-align: center;
        margin-bottom: 30px;
    }
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    ::-webkit-scrollbar-thumb {
        background: #4a69bb;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("📚 RAG Chatbot")
    st.markdown("#### Ask questions about the SQuAD dataset")
    
    if "chatbot" not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            st.session_state.chatbot = RAGChatbot("squadnew.xlsx")
            st.success("Chatbot initialized successfully!")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm a RAG-based chatbot. Ask me any question based on the SQuAD dataset."}
        ]
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.container():
            st.markdown(f"""
            <div class="chat-message {message['role']}">
                <div class="message">
                    {message['content']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # User input
    if prompt := st.chat_input("Ask a question"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.container():
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message">
                    {prompt}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.chatbot.ask(prompt)
                
                # Display assistant message
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="message">
                            {response}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Add assistant message to history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response
                })
                
            except Exception as e:
                error_message = f"Sorry, I encountered an error: {str(e)}"
                
                with st.container():
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="message">
                            {error_message}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": error_message
                })

if __name__ == "__main__":
    main()