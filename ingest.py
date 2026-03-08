import os
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

model = SentenceTransformer('all-MiniLM-L6-v2')

conn = psycopg2.connect(
    dbname="ragdb",
    user="postgres",
    password="postgres",
    host="localhost",
    port="5432"
)

register_vector(conn)

cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT,
    chunk_index INTEGER,
    content TEXT,
    embedding VECTOR(384)
)
""")

cursor.execute("""
CREATE INDEX IF NOT EXISTS documents_embedding_idx
ON documents
USING hnsw (embedding vector_cosine_ops);
""")

cursor.execute("DELETE FROM documents;")

folder = "documents"

for file in os.listdir(folder):

    with open(f"{folder}/{file}") as f:
        text = f.read()

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk)

        cursor.execute(
            """
            INSERT INTO documents (filename, chunk_index, content, embedding)
            VALUES (%s, %s, %s, %s)
            """,
            (file, i, chunk, embedding.tolist())
        )

conn.commit()
cursor.close()
conn.close()

print("Documents embedded successfully.")