# 📚 RAG Chatbot 🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A Retrieval-Augmented Generation (RAG) Chatbot built with Streamlit, LangChain, and HuggingFace. This chatbot answers questions based on the SQuAD dataset using a combination of retrieval and generation techniques. It features a modern, user-friendly interface with a sleek dark theme and responsive design.

---

## ✨ Features

- **RAG Architecture**: Combines retrieval (FAISS vectorstore) with generation (HuggingFace LLM) for accurate answers.
- **Modern UI**: Built with Streamlit, featuring a dark theme, chat bubbles, and smooth animations.
- **Efficient Retrieval**: Uses sentence embeddings to retrieve relevant context from the SQuAD dataset.
- **Concise Answers**: Provides direct answers without repeating the question or context.
- **Error Handling**: Gracefully handles errors with user-friendly messages.
- **Customizable**: Easily modify the dataset, embeddings, or LLM model as needed.

---

## 🛠️ Installation

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (optional, for cloning the repository)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/rag-chatbot.git
   cd rag-chatbot
