import os

from fastapi import FastAPI
from google import genai
from google.genai import types
import httpx
from starlette.requests import Request

from question_answer import QuestionAnswer
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))


@app.post("/")
async def root(request: Request):
    pdf_byte = await request.body()
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
            prompt])
    print(response.text)
    return response.parsed
