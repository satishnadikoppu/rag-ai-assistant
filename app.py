from fastapi import FastAPI
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

register_vector(conn)


# --- Agent Tools ---

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def search_documents(query: str) -> str:
    embedding = model.encode(query)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT filename, chunk_index, content
        FROM documents
        ORDER BY embedding <-> %s::vector
        LIMIT 7
    """, (embedding.tolist(),))
    results = cursor.fetchall()

    if not results:
        return "No relevant documents found."

    parts = []
    for filename, chunk_index, content in results:
        parts.append(f"[Source: {filename} | Chunk: {chunk_index}]\n{content}")

    return "\n\n".join(parts)


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns the current system time.",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": "Searches the document knowledge base and returns relevant chunks for a given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up in the documents."
                    }
                },
                "required": ["query"]
            }
        }
    }
]


TOOL_REGISTRY = {
    "get_current_time": lambda _: get_current_time(),
    "search_documents": lambda args: search_documents(args["query"]),
}


def run_agent(question: str) -> dict:
    messages = [{"role": "user", "content": question}]
    tool_calls_log = []

    # Round 1: LLM decides whether to call a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=TOOLS,
        messages=messages
    )

    msg = response.choices[0].message

    # If the model wants to use a tool
    if msg.tool_calls:
        messages.append(msg)  # add assistant message with tool_calls

        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_input = json.loads(tc.function.arguments)
            tool_result = TOOL_REGISTRY[tool_name](tool_input)

            print(f"[Agent] Tool called: {tool_name} | Input: {tool_input}")
            tool_calls_log.append({"tool": tool_name, "input": tool_input})

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result
            })

        # Round 2: send tool results back for final answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            tools=TOOLS,
            messages=messages
        )
        msg = response.choices[0].message

    return {"answer": msg.content, "tools_used": tool_calls_log}


@app.get("/agent")
def agent(question: str):
    return run_agent(question)


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