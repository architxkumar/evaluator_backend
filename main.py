import os
import json
import pandas as pd
from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Dict, List, Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
from google.genai import types
import httpx
from starlette.requests import Request


class QuestionAnswer(BaseModel):
    question: str
    answer: str

load_dotenv()
app = FastAPI()
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))


# -------------------------------
# Evaluation Functions
# -------------------------------
def load_chunks(folder="knowledge_chunks"):
    chunks = []
    if not os.path.exists(folder):
        print(f"⚠️ Warning: Knowledge chunks folder '{folder}' not found.")
        return chunks

    for file_name in sorted(os.listdir(folder)):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder, file_name), "r", encoding="utf-8") as f:
                chunks.append(f.read())
    print(f"✅ Loaded {len(chunks)} chunks from {folder}")
    return chunks


def init_azure_client():
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise ValueError("❌ GITHUB_TOKEN not found in environment variables.")

    endpoint = "https://models.github.ai/inference"
    model = "openai/gpt-4o-mini"
    azure_client = ChatCompletionsClient(endpoint=endpoint, credential=AzureKeyCredential(token))
    return azure_client, model


def build_faiss_index(chunks, model_name="all-MiniLM-L6-v2", emb_file="chunk_embeddings.npy", index_file="faiss.index"):
    if not chunks:
        print("⚠️ Warning: No chunks provided, creating empty index.")
        model = SentenceTransformer(model_name)
        embeddings = np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        return index, embeddings, model

    model = SentenceTransformer(model_name)
    new_chunks = []

    if os.path.exists(emb_file) and os.path.exists(index_file):
        print("✅ Loading saved embeddings and FAISS index...")
        embeddings = np.load(emb_file)
        index = faiss.read_index(index_file)
        existing_chunk_count = embeddings.shape[0]
        if len(chunks) > existing_chunk_count:
            new_chunks = chunks[existing_chunk_count:]
        else:
            return index, embeddings, model
    else:
        embeddings = np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")
        index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        new_chunks = chunks

    if new_chunks:
        print(f"Embedding {len(new_chunks)} new chunks...")
        new_embeddings = np.array([model.encode(chunk) for chunk in tqdm(new_chunks)]).astype("float32")
        embeddings = np.vstack([embeddings, new_embeddings])
        index.add(new_embeddings)

        np.save(emb_file, embeddings)
        faiss.write_index(index, index_file)
        print(f"✅ Updated embeddings saved to {emb_file}, FAISS index saved to {index_file}")

    return index, embeddings, model


def retrieve_relevant_chunks_semantic(question, chunks, index, embeddings, sbert_model, top_k=3):
    if not chunks or index.ntotal == 0:
        return []

    q_vec = sbert_model.encode(question).astype("float32")
    D, I = index.search(np.array([q_vec]), min(top_k, len(chunks)))
    top_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    return top_chunks


def evaluate_answer(question, student_answer, relevant_chunks, azure_client, model):
    student_answer = " ".join(student_answer.splitlines())
    context = "\n\n".join(relevant_chunks) if relevant_chunks else "No relevant reference material found."

    prompt = f"""
You are a strict examiner grading a student's answer.

Reference Material (may be incomplete):
{context}

Question: {question}
Student Answer: {student_answer}

Instructions:
1. Check if the student's answer is correct based on the reference material.
2. If the reference material doesn't fully cover the question, use your general knowledge to evaluate correctness.
3. Perform semantic understanding: even if the question wording doesn't exactly match the reference material, still find the relevant concepts.
4. Give a score from 0 to 5:
   - 0 = completely wrong
   - 5 = perfect
5. Provide a short feedback explaining the score.
6. List all key concepts or terms in the student's answer that match the reference material.

Respond ONLY in valid JSON format like:
{{ 
    "score": X, 
    "feedback": "...",
    "concepts": ["concept1", "concept2"]
}}
"""

    try:
        response = azure_client.complete(
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            model=model
        )
        raw_content = response.choices[0].message["content"]
        print("RAW GPT RESPONSE:", raw_content)

        json_start = raw_content.find("{")
        json_end = raw_content.rfind("}") + 1
        json_text = raw_content[json_start:json_end]
        return json.loads(json_text)

    except Exception as e:
        print("⚠️ Error parsing GPT response, returning default:", e)
        return {"score": 0, "feedback": "Could not evaluate answer.", "concepts": []}


# -------------------------------
# Main FastAPI Endpoint
# -------------------------------
@app.post("/")
async def root(request: Request):
    """Extract Q&A pairs from PDF and evaluate them - ONE STOP SHOP!"""
    try:
        # Get PDF bytes
        pdf_byte = await request.body()

        # Step 1: Extract Q&A pairs using Gemini
        print("📄 Extracting Q&A pairs from PDF...")
        prompt = "Extract all the content from the PDF and if you aren't able to extract text, say 'no text found' and then arrange the contents as per the response schema."

        response = client.models.generate_content(
            config={
                "response_mime_type": "application/json",
                "response_schema": list[QuestionAnswer],
            },
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=pdf_byte,
                    mime_type='application/pdf',
                ),
                prompt
            ]
        )

        extracted_qa_pairs = response.parsed
        print(f"✅ Extracted {len(extracted_qa_pairs)} Q&A pairs")

        if not extracted_qa_pairs:
            return {
                "success": False,
                "error": "No question-answer pairs found in PDF",
                "results": [],
                "summary": {"total_score": 0, "max_score": 0, "average_score": 0.0}
            }

        # Step 2: Load knowledge chunks
        print("📚 Loading knowledge chunks...")
        chunks = load_chunks("knowledge_chunks")

        # Step 3: Build FAISS index
        print("🔍 Building semantic search index...")
        index, embeddings, sbert_model = build_faiss_index(chunks)

        # Step 4: Initialize Azure client
        print("🤖 Initializing AI evaluator...")
        azure_client, model = init_azure_client()

        # Step 5: Evaluate each Q&A pair
        print("📊 Evaluating answers...")
        results = []
        for qa_pair in extracted_qa_pairs:
            question = qa_pair.question if hasattr(qa_pair, 'question') else str(qa_pair.get('question', ''))
            answer = qa_pair.answer if hasattr(qa_pair, 'answer') else str(qa_pair.get('answer', ''))

            # Get relevant context chunks
            top_chunks = retrieve_relevant_chunks_semantic(question, chunks, index, embeddings, sbert_model, top_k=3)

            # Evaluate the answer
            eval_result = evaluate_answer(question, answer, top_chunks, azure_client, model)

            results.append({
                "question": question,
                "answer": answer,
                "score": eval_result.get("score", 0),
                "feedback": eval_result.get("feedback", ""),
                "concepts": eval_result.get("concepts", [])
            })

        # Step 6: Calculate summary
        total_score = sum(r["score"] for r in results)
        max_score = len(results) * 5
        average_score = total_score / len(results) if results else 0.0

        print(f"✅ Evaluation complete! Average score: {average_score:.2f}/5.0")

        return {
            "success": True,
            "results": results,
            "summary": {
                "total_score": total_score,
                "max_score": max_score,
                "average_score": round(average_score, 2),
                "total_questions": len(results)
            }
        }

    except Exception as e:
        print(f"❌ Error in processing: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "results": [],
            "summary": {"total_score": 0, "max_score": 0, "average_score": 0.0}
        }