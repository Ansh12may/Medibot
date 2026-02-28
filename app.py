from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os



app = Flask(__name__, template_folder="tempelates")



load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")



embeddings = download_hugging_face_embeddings()


index_name = "medicalbot"

docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.4,
    max_tokens=500
)


system_prompt = (
    "You are a medical question-answering assistant. "
    "Only answer medical-related questions (diseases, symptoms, treatment, anatomy, diagnosis). "
    "If the question is not medical, respond exactly with: "
    "'Sorry, I can only answer medical-related questions.' "
    "Use the retrieved context to answer. "
    "If the answer is not in the context, say you don't know. "
    "Use three sentences maximum.\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "input": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)



@app.route("/")
def index():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    response = rag_chain.invoke(msg)

    print("Bot:", response)
    return response



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)