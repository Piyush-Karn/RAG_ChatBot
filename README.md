# 📚 SQuAD RAG Chatbot

<div align="center">

![GitHub stars](https://img.shields.io/github/stars/piyush-karn/RAG_ChatBot?style=social)
![GitHub forks](https://img.shields.io/github/forks/piyush-karn/RAG_ChatBot?style=social)
![GitHub issues](https://img.shields.io/github/issues/piyush-karn/RAG_ChatBot)
![License](https://img.shields.io/github/license/piyush-karn/RAG_ChatBot)

**An elegant, interactive chatbot powered by Retrieval-Augmented Generation (RAG) to answer questions about the SQuAD dataset**

[Features](#features) • [Demo](#demo) • [Installation](#installation) • [Usage](#usage) • [How it Works](#how-it-works) • [Contributing](#contributing) • [License](#license)

<img src="https://raw.githubusercontent.com/piyush-karn/RAG_ChatBot/main/assets/chatbot-preview.png" width="600" alt="SQuAD RAG Chatbot Preview">

</div>

## ✨ Features

- 🚀 **Retrieval-Augmented Generation** - Combines vector search with language models for accurate answers
- 💬 **Interactive Chat Interface** - Beautiful, responsive design built with Streamlit
- 🧠 **Local LLM Support** - Uses Facebook's OPT-125M model for generating responses
- 📊 **SQuAD Dataset Integration** - Pre-loaded with Stanford Question Answering Dataset
- 📝 **Context-Aware Responses** - Provides answers based on retrieved relevant contexts
- 🔍 **Vector Search** - Fast similarity search using Facebook AI Similarity Search (FAISS)
- 💾 **Persistent Vector Storage** - Saves embeddings to avoid regeneration on restart

## 🎮 Demo

<div align="center">
  <img src="https://raw.githubusercontent.com/yourusername/squad-rag-chatbot/main/assets/chatbot-demo.gif" width="700" alt="Chatbot Demo">
</div>

## 🛠️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/squad-rag-chatbot.git
cd squad-rag-chatbot
```

2. **Create and activate a virtual environment**

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the SQuAD dataset**
   
The application expects an Excel file named `squadnew.xlsx` in the root directory. You can download the SQuAD dataset and convert it to Excel format using the included utility:

```bash
python utils/download_squad.py
```

## 🚀 Usage

Run the Streamlit application:

```bash
streamlit run str.py
```

The application will be available at `http://localhost:8501` in your web browser.

## 🧩 How it Works

### Architecture

```
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   SQuAD     │         │ Vector      │         │  Language   │
│  Dataset    │────────▶│   Store     │────────▶│   Model     │
└─────────────┘         └─────────────┘         └─────────────┘
       │                       ▲                       │
       │                       │                       │
       ▼                       │                       ▼
┌─────────────┐         ┌─────────────┐         ┌─────────────┐
│   Text      │         │  Retrieval  │         │  Response   │
│  Splitter   │────────▶│    Engine   │◀────────│ Generation  │
└─────────────┘         └─────────────┘         └─────────────┘
                                ▲                       │
                                │                       │
                                ▼                       ▼
                        ┌───────────────────────────────┐
                        │       Streamlit UI            │
                        └───────────────────────────────┘
```

### Components

1. **Data Processing**
   - Loads SQuAD dataset from Excel
   - Splits text into manageable chunks using RecursiveCharacterTextSplitter

2. **Embedding & Storage**
   - Uses `sentence-transformers/all-MiniLM-L6-v2` for generating embeddings
   - Stores vector embeddings using FAISS for efficient similarity search
   - Caches vectors to disk to improve startup performance

3. **Language Model**
   - Employs Facebook's OPT-125M model for generating responses
   - Configured with appropriate parameters for conversational output

4. **RAG Pipeline**
   - Retrieves relevant context for user queries
   - Provides context to the language model for informed answers
   - Uses a custom prompt template to guide response generation

5. **User Interface**
   - Responsive chat interface with Streamlit
   - Custom CSS for improved visual aesthetics
   - Session state management for conversation history

## 📋 Requirements

```
pandas
torch
streamlit>=1.24.0
langchain>=0.0.267
faiss-cpu>=1.7.4
transformers>=4.30.2
sentence-transformers>=2.2.2
openpyxl>=3.1.2
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- [Stanford Question Answering Dataset (SQuAD)](https://rajpurkar.github.io/SQuAD-explorer/)
- [LangChain](https://github.com/hwchase17/langchain)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Facebook AI Similarity Search (FAISS)](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

---

<div align="center">
  Made with ❤️ by <a href="https://github.com/yourusername">YourUsername</a>
</div>
