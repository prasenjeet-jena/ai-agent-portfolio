# Import the tool that helps us find and load our hidden passwords from the .env file
from dotenv import load_dotenv, find_dotenv

# Import the LangChain tool that lets our script connect directly to OpenAI's AI models
from langchain_openai import ChatOpenAI

# Import the LangChain tool that helps us create a fill-in-the-blank instruction sheet
from langchain_core.prompts import ChatPromptTemplate

# Import the LangChain tool that cleans up the AI's response into readable normal text
from langchain_core.output_parsers import StrOutputParser

# 1. Automatically search for the hidden .env file and securely load our secret API keys into memory
load_dotenv(find_dotenv())

# 2. Create our AI connection and tell it specifically to use the cost-effective "gpt-4o-mini" model.
# 3. We use temperature=0 because this controls how "creative" the AI is.
# A temperature of 0 means the AI is completely robotic, strict, factual, and not allowed to "make things up".
# This is perfect for classification tasks where we need exact output with no conversational fluff!
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 2. Create a reusable instruction template that sets the rules for how the AI sorts user feedback.
# 4. We specifically instruct the AI to return ONLY the exact category name and absolutely nothing else.
prompt = ChatPromptTemplate.from_template(
    "You are an AI assistant that sorts product feedback.\n\n"
    "Classify the following user feedback into exactly ONE of these categories:\n"
    "- bug_report\n"
    "- feature_request\n"
    "- complaint\n"
    "- praise\n"
    "- question\n\n"
    "CRITICAL RULE: Return ONLY the exact category name. Nothing else. No explanation, no punctuation.\n\n"
    "Feedback that needs to be sorted: {feedback}"
)

# Set up a final filter that works to take only the text string out of the AI's complex response data
output_parser = StrOutputParser()

# Build our "chain" (assembly line) using LangChain's LCEL pipe (|) syntax
# It pushes data forward left to right: format the prompt -> ask the AI -> clean the text output
classification_chain = prompt | llm | output_parser

# 5. Create a list of 5 test inputs (different pieces of user feedback) to see if our classifier works
test_inputs = [
    "The app crashes every time I upload a file",
    "It would be great if you added dark mode",
    "Your support team is absolutely terrible",
    "This product has completely transformed how our team works",
    "How do I export my data to CSV?,
    "I've been waiting 3 weeks for my refund and nobody is responding",
    "The new update completely broke my existing workflow",
    "Why did you remove the feature I use every day?"
]

# Print a nice header so we know our automated tests are starting
print("\n--- Starting Feedback Classification Chain Test ---\n")

# Loop through each piece of feedback inside our list of test inputs one by one
for feedback_item in test_inputs:
    # Trigger the chain (assembly line) by plugging the current feedback text into the {feedback} blank space
    result_category = classification_chain.invoke({"feedback": feedback_item})
    
    # 6. Print the original feedback text we sent, neatly aligned
    print(f"Feedback: {feedback_item}")
    
    # 6. Print the category the AI confidently decided it belonged to
    print(f"Category: {result_category}\n")

# Print a nice footer so we know our test is finalized
print("---------------------------------------------------\n")
