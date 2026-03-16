# Import the tool that helps us find and load our hidden passwords from the .env file
from dotenv import load_dotenv, find_dotenv

# Import the LangChain tool that lets our script connect directly to OpenAI's AI models
from langchain_openai import ChatOpenAI

# Import the LangChain tool that helps us create a fill-in-the-blank instruction sheet
from langchain_core.prompts import ChatPromptTemplate

# Import Pydantic tools. Pydantic is a very strict "bouncer" for our data.
# In AI agent design, AIs naturally want to talk in free-flowing paragraphs.
# Pydantic forces the AI to reply ONLY in a very rigid, specific structure (like a clean spreadsheet).
# For a PM, this matters immensely: it turns unpredictable AI "chat" into clean, structured data
# that can be reliably fed directly into databases, dashboards, or other software systems!
from pydantic import BaseModel, Field

# 1. Automatically search for the hidden .env file and securely load our secret API keys into memory
load_dotenv(find_dotenv())

# 2. Create our AI connection using the cost-effective "gpt-4o-mini" model.
# We keep temperature low (0) so the AI acts like a strict data analyst, not a creative writer.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 4. Define our exact Output Schema using Pydantic.
# This strictly defines the "blueprint" of the JSON object we demand the AI to return.
class ReviewSummary(BaseModel):
    # We demand exactly one of these three words
    sentiment: str = Field(description="Must be exactly: positive, negative, or neutral")
    # We demand an integer number between 1 and 10
    score: int = Field(description="A score from 1 to 10")
    # We demand a list of bullet points (up to 3)
    key_issues: list[str] = Field(description="List of up to 3 main issues mentioned")
    # We demand exactly one concrete proposed action
    recommended_action: str = Field(description="One specific action the PM team should take")
    # We demand exactly one priority level
    priority: str = Field(description="Must be exactly: high, medium, or low")

# 3. We use a special LangChain method (.with_structured_output).
# This forcefully binds our strict Pydantic blueprint to the AI.
# The AI must now obey this structure or the code will crash.
structured_llm = llm.with_structured_output(ReviewSummary)

# Create an instruction template to feed the raw text to our strict AI
prompt = ChatPromptTemplate.from_template(
    "You are an expert Product Manager assistant.\n"
    "Analyze the following product review and extract the required structured information.\n\n"
    "Review: {review}"
)

# Build our assembly line: format the prompt -> send it to the structured AI
summarizer_chain = prompt | structured_llm

# 5. Create a list of 3 test reviews to see our structured data extractor in action
test_reviews = [
    "The app is great but crashes on iOS 16 and the export feature is completely broken. Been waiting 3 months for a fix.",
    "Absolutely love the new dashboard. Clean, fast, intuitive. Best update yet.",
    "Setup took forever, docs are outdated, and support never responds. Considering switching to a competitor."
]

import json

# Print a nice header so we know our automated tests are starting
print("\n--- Starting Review Summarizer Test ---\n")

# Loop through each test review one by one
for review in test_reviews:
    # Trigger the chain: analyze the review and rigorously force the output into our Pydantic blueprint
    result = summarizer_chain.invoke({"review": review})
    
    # Print the raw text we sent
    print(f"Original Review:\n\"{review}\"")
    
    # 6. Print the incredibly clean, structured, and predictable JSON output!
    # (The .model_dump() turns the Pydantic object back into standard Python data,
    # and json.dumps makes it visually pretty and indented for our terminal)
    print("Structured Output:")
    print(json.dumps(result.model_dump(), indent=2))
    print("\n-------------------------\n")
