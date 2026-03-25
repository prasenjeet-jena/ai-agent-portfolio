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

# 2. Create our AI connection and tell it specifically to use the cost-effective "gpt-4o-mini" model
llm = ChatOpenAI(model="gpt-4o-mini")

# 3. Create a reusable instruction template that leaves a blank space exactly named {question}
prompt = ChatPromptTemplate.from_template("{question}")

# Set up a final filter that works to take only the text string out of the AI's complex response data
output_parser = StrOutputParser()

# 4. Build our "chain" (assembly line) using LangChain's LCEL pipe (|) syntax
# It pushes data forward left to right: format the prompt -> ask the AI -> clean the text output
chain = prompt | llm | output_parser

# 5. Save the exact question we want to ask into a designated text variable
my_question = "What are the 3 most important product decisions a PM makes when building an AI agent?"

# Trigger the chain (assembly line) by plugging our actual question text into the {question} blank space
response = chain.invoke({"question": my_question})

# 6. Finally, print out our AI's final cleaned-up text response on the terminal screen!
print("\n--- AI Response ---\n")
print(response)
print("\n-------------------\n")
