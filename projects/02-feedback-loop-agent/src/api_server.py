from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Load environment variables (API keys)
load_dotenv(find_dotenv())

app = FastAPI(title="Feedback PRD Generator API")

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the LLM (gpt-4o-mini is fast and capable for this task)
# The API key is automatically picked up from the environment via OPENAI_API_KEY
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

class PRDRequest(BaseModel):
    theme: str
    summary: str
    raw_feedback_items: List[Dict[str, Any]]

@app.post("/generate-prd")
async def generate_prd(request: PRDRequest):
    """
    Generates a highly technical PRD based on a set of feedback items.
    """
    
    system_prompt = SystemMessage(content=(
        "Act as a Senior AI Product Manager at a Tier-1 Tech company. "
        "Analyze the provided 10-15 user feedback items. "
        "Write a highly technical and professional PRD. "
        "Include: Problem Statement, hypothesis, requirements in detail or Proposed Solution, "
        "Acceptance criteria and Success Metrics (KPIs) and long user stories"
    ))

    # Format the raw feedback items into a readability string for the LLM
    feedback_text_list = []
    for item in request.raw_feedback_items:
        text = item.get("text", "")
        if text:
             feedback_text_list.append(f"- {text}")
    
    formatted_feedback = "\n".join(feedback_text_list)

    human_prompt = HumanMessage(content=(
        f"Theme: {request.theme}\n"
        f"Summary: {request.summary}\n\n"
        f"Raw User Feedback:\n{formatted_feedback}\n\n"
        "Please generate the PRD in Markdown format."
    ))

    # Invoke the LLM
    response = llm.invoke([system_prompt, human_prompt])
    
    # Return the generated content as a clean string
    return response.content
