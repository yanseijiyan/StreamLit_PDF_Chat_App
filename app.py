import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import gc

# Set up the OpenAI API key
os.environ['OPENAI_API_KEY'] = 'Your-OpenAI-API-Key'

def main():
    # Application header
    st.header('Chat with PDF 💬')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown('''
    This is an LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://platform.openai.com/docs/models) LLM Model 
    ''')

    # Upload a PDF File
    pdf = st.file_uploader("Upload your PDF File", type='pdf')

    if pdf is not None:
        # Read the content of the PDF
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        st.write(text)

        # Split the PDF text into chunks for processing
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # Use the PDF file name as an identifier to save/retrieve embeddings
        store_name = pdf.name[0]
        st.write(store_name)

        # Check if embeddings are already saved for this PDF
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the Disk')
        else:
            # Create embeddings for the PDF text chunks
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Created')

        # Query input to ask questions about the PDF file
        query = st.text_input("Ask Question from your PDF File")
        if query:
            # Perform similarity search on the documents using embeddings
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm=llm, chain_type='stuff')

            # Execute the OpenAI question-answering model
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query, max_tokens=1000)
                print(cb)
            st.write(response)

pdf_reader = None
VectorStore = None
gc.collect()

if __name__ == '__main__':
    main()