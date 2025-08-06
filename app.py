import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import time
import pandas as pd

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

#read pdf, go through each page and extract text
def get_pdf_text(pdf_docs):
    text = ""
    doc_times = []
    for pdf in pdf_docs:
        start = time.time()
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
        end = time.time()
        doc_times.append((pdf.name, end - start))
    return text, doc_times

#split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks

# get embeddings and create a vector store and save it locally
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = 'models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

# create a prompt template for the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. 
    If the answer is not in the provided context, just say "Answer is not available in the context". 
    Don't provide the wrong answer. 

    Context: 
    {context}

    Question: 
    {question}

    Answer:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)

    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# function to handle user input and return the response
def user_input(user_question):
    st.info("⏳ Generating response...")

    start_time = time.time()

    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    faiss_start = time.time()
    docs = new_db.similarity_search(user_question)
    faiss_end = time.time()
    st.write(f"🔍 Similarity search took {faiss_end - faiss_start:.2f} seconds")

    chain = get_conversational_chain()

    llm_start = time.time()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    llm_end = time.time()

    st.write(f"🧠 LLM inference took {llm_end - llm_start:.2f} seconds")
    total_time = time.time() - start_time

    st.write(f"⚡ Total time taken: {total_time:.2f} seconds")
    st.write("### Answer:")
    st.write(response['output_text'])

#  log metrics to csv file

LOG_FILE = "metrics_log.csv"
def log_metrics(metrics_dict):
    df = pd.DataFrame([metrics_dict])
    if os.path.exists(LOG_FILE):
        df.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_FILE, index=False)

# function to set up the Streamlit app
def main():
    st.set_page_config("CHAT with multiple PDFs")
    st.header("Chat with multiple PDFs using Gemini")

    user_question = st.text_input("Ask a question about the PDFs")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)  
        if st.button("Submit and Process"):
            with st.spinner("Processing PDFs..."):
                overall_start = time.time()

                raw_text, doc_times = get_pdf_text(pdf_docs)
                st.write("✅ Extracted text from PDFs")

                text_chunks = get_text_chunks(raw_text)
                st.write(f"🧩 Total chunks created: {len(text_chunks)}")

                embedding_start = time.time()
                get_vector_store(text_chunks)
                embedding_end = time.time()

                overall_end = time.time()

                total_time = overall_end - overall_start
                embedding_time = embedding_end - embedding_start

                # Log and display per-document times
                for doc_name, doc_time in doc_times:
                    st.write(f"📄 {doc_name} processed in {doc_time:.2f} sec")

                # Save metrics to CSV
                log_metrics({
                    "timestamp": pd.Timestamp.now(),
                    "type": "upload",
                    "num_pdfs": len(pdf_docs),
                    "total_chunks": len(text_chunks),
                    "embedding_time_sec": round(embedding_time, 2),
                    "total_processing_time_sec": round(total_time, 2)
                })

                st.success(f"Done! Processing took {total_time:.2f} seconds")

if __name__ == "__main__":
    main()
