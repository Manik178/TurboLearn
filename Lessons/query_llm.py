import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def generate_lesson(topic: str, level: str, context: str) -> str:
    prompt = f"""
    You are a helpful teacher creating a lesson.

    Topic: {topic}
    Student level: {level}

    Context from textbook:
    {context}

    Generate a structured lesson with:
    1. Introduction
    2. Detailed explanation tailored to {level} students
    3. Real-world examples
    4. Practice questions with answers
    """

    resp = genai.GenerativeModel("gemini-1.5-flash").generate_content(prompt)
    return resp.text
