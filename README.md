# 📄 Chat with Multiple PDFs using Gemini (Google Generative AI)

This Streamlit web app allows users to upload and interact with multiple PDF documents using natural language. It uses Google’s **Gemini Pro** model for intelligent question-answering, and FAISS for semantic search over PDF content.

---

## 🚀 Features

- Upload multiple PDFs and extract their text
- Split large text into manageable chunks
- Generate vector embeddings using Google Gemini
- Store and search documents using FAISS vector store
- Ask any question about the content — get detailed, contextual answers
- Secure and fast — runs locally!

---

## 🧩 Tech Stack

- `Streamlit` – UI Framework
- `LangChain` – Chain and Prompt orchestration
- `Google Generative AI` – Gemini Pro + Embedding API
- `FAISS` – Vector store for similarity search
- `PyPDF2` – PDF parsing

---

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/gemini-multi-pdf-chat.git
cd gemini-multi-pdf-chat
```

### 2. Set up a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```


### 🔑 Setup Google API Key
Get your API key from Google AI Studio.

Create a .env file in the project root:
```ini
GOOGLE_API_KEY=your_api_key_here
```

### ▶️ Run the App
```bash
streamlit run app.py
```


### 🧪 How It Works
1. Upload one or more PDFs from the sidebar.

2. The app:

  - Extracts all text from PDFs

  - Splits text into chunks

  - Generates vector embeddings with Gemini

  - Stores them in FAISS

3. Ask a question in the input box.

4. The system retrieves the most relevant chunks and passes them to Gemini to generate an answer.

### 📁 Project Structure

```bash
gemini-multi-pdf-chat/
│
├── app.py                 # Main Streamlit application
├── faiss_index/           # Folder to store FAISS index (auto-created)
├── .env                   # Contains your Google API Key (not committed)
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

### 📌 To-Do (Optional Enhancements)
 - Add chat history and memory

 - Tag chunks by PDF source for better traceability

 - Deploy on Streamlit Cloud or Render

 - Add UI feedback for errors and loading

### 📜 License
This project is open-source and free to use under the MIT License.

### 🙌 Acknowledgements
- LangChain

- Google Generative AI

- Streamlit

- FAISS

### 💡 Author
####  Atharva Rajadhyaksha 🔗 [LinkedIn](https://www.linkedin.com/in/atharva-rajadhyaksha)

### Demo
[Here](https://chat-with-multiple-pdf-gemini.streamlit.app/)
