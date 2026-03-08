from fastapi import FastAPI
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
import anthropic
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

# client = anthropic.Anthropic(
#     api_key=os.getenv("ANTHROPIC_API_KEY")
# )

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

register_vector(conn)

@app.get("/ask")

def ask(question: str):

    embedding = model.encode(question)

    cursor = conn.cursor()

    cursor.execute("""
    SELECT filename, chunk_index, content
    FROM documents
    ORDER BY embedding <-> %s::vector
    LIMIT 7
    """, (embedding.tolist(),))

    results = cursor.fetchall()
    sources = []

    print("Retrieved chunks:")

    for filename, chunk_index, content in results:
        print(f"{filename} | chunk {chunk_index}")
        sources.append({
            "filename": filename,
            "chunk_index": chunk_index
        })

    #context = "\n".join([d[0] for d in docs])

    #context = "\n\n".join([r[2] for r in docs])
    context_parts = []

    for filename, chunk_index, content in results:
        context_parts.append(
            f"[Source: {filename} | Chunk: {chunk_index}]\n{content}"
        )

    context = "\n\n".join(context_parts)

    prompt = f"""
    You are a question-answering assistant.

    Use ONLY the information from the context below to answer the question.
    You may combine information from multiple context chunks to form the answer.

    If the answer cannot be determined from the context, reply:
    "I don't know based on the provided documents."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    # response = client.messages.create(
    #     model="claude-3-haiku-20240307",
    #     max_tokens=300,
    #     messages=[
    #         {"role": "user", "content": prompt}
    #     ]
    # )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    #return {"answer": response.content[0].text}
    return {
        "answer": response.choices[0].message.content,
        "sources": sources
    }