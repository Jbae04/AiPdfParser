from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

def create_ai_agent(vector_store, api_key):
    llm = OpenAI(openai_api_key=api_key, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def ask_question(qa_chain, question):
    response = qa_chain.run(question)
    return response