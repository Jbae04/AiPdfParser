from pdf_parser import extract_text_from_pdf
from text_processor import split_text_into_chunks
from vector_store import create_vector_store
from ai_agent import create_ai_agent, ask_question

def main(pdf_path, api_key):
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    
    # Split text into chunks
    chunks = split_text_into_chunks(text)
    
    # Create vector store
    vector_store = create_vector_store(chunks, api_key)
    
    # Create AI agent
    qa_chain = create_ai_agent(vector_store, api_key)
    
    # Ask questions
    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        answer = ask_question(qa_chain, question)
        print(f"Answer: {answer}\n")

if __name__ == "__main__":
    pdf_path = "../data/test.pdf"  # Path to your PDF file
    api_key = "ASDASDASD" # Replace with your OpenAI API key
    main(pdf_path, api_key)