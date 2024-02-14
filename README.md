# LLM-PDF-ChatApp

LLM-PDF-ChatApp is a Streamlit web application powered by LangChain and OpenAI's Language Model (LLM) that allows users to upload a PDF file, ask questions related to the content of the PDF, and receive answers generated by the language model.

## Features

- **PDF Upload:** Users can upload a PDF file containing text content.
- **Question-Answering:** Users can ask questions related to the content of the uploaded PDF.
- **Language Model Integration:** Utilizes OpenAI's Language Model (LLM) for generating answers to user queries.
- **Embeddings Storage:** Stores embeddings of PDF text chunks for efficient question-answering.

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yanseijiyan/StreamLit_PDF_Chat_App.git
    ```

2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up your OpenAI API key by replacing `'Your-OpenAI-API-Key'` in `app.py` with your actual API key.

4.Replace 'Your-OpenAI-API-Key' with your actual OpenAI API key in the code.


5. Run the application:

    ```bash
    streamlit run app.py
    ```

## Usage

1. Upload a PDF file using the file uploader.
2. Ask questions related to the content of the PDF in the provided text input.
3. View the answers generated by the language model.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please feel free to open an issue or create a pull request.
