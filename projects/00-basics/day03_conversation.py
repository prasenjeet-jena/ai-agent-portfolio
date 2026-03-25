# Import the tool that helps us find and load our hidden passwords from the .env file
from dotenv import load_dotenv, find_dotenv

# Import the LangChain tool that lets our script connect directly to OpenAI's AI models
from langchain_openai import ChatOpenAI

# Import the LangChain tools needed to build our conversation structure
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Import the tools needed for the AI's "memory".
# Memory in AI is a crucial concept for a Product Manager to understand. 
# AI models like GPT-4 actually have ZERO memory by default (they are "stateless"). 
# Every single time you send a message, the AI forgets who you are and what you said 2 seconds ago.
# Therefore, "memory" is simply the act of re-sending the ENTIRE past conversation history BACK to the AI 
# every single time you ask a new question, so it can re-read what was just said and pretend it remembers!
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 1. Automatically search for the hidden .env file and securely load our secret API keys into memory
load_dotenv(find_dotenv())

# 3. Create our AI connection using "gpt-4o-mini".
# We use temperature=0.7 to give the AI a bit of creative freedom to act like a human advisor. 
# 0.7 means it won't be as strictly robotic as temperature=0, but also won't hallucinate wildly.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# 4. Create an instruction template setting up the AI's personality and rules.
# We also leave a crucial blank space (MessagesPlaceholder) where LangChain will inject all our past chat history.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Product Manager advisor. Give concise, extremely direct, and actionable advice. No fluff."),
    
    # This acts as a permanent placeholder slot! Every time we talk to the AI, 
    # LangChain dynamically shoves the past 5 messages into this slot so the AI can "remember" context.
    MessagesPlaceholder(variable_name="history"),
    
    # This slot holds our brand new message we want to ask right now!
    ("human", "{human_input}")
])

# Build our core assembly line: format the prompt -> ask the AI -> clean the text output
chain = prompt | llm | StrOutputParser()

# Create an empty, temporary "database" in our computer's memory to store our chat history logs
memory_database = InMemoryChatMessageHistory()

# Explain to LangChain how to fetch and save messages to our temporary "database"
def get_session_history(session_id: str):
    # For this simple script, we just return our one shared memory database for everything
    return memory_database

# Wrap our core assembly line with a special tool (RunnableWithMessageHistory).
# This tool automatically intercepts all our inputs/outputs and saves them to the memory database for us!
conversational_agent = RunnableWithMessageHistory(
    chain,
    # Tell it exactly how to find our memory database
    get_session_history,
    # Tell it exactly which blank space in the prompt the user's new question goes into
    input_messages_key="human_input",
    # Tell it exactly which blank space in the prompt the past history goes into
    history_messages_key="history",
)

# 5. Create a list of 5 deeply connected questions to prove the AI requires "memory" to answer them properly
test_conversation = [
    "I am a PM building an AI agent for enterprise onboarding. What's the most important thing to get right?",
    "Can you elaborate on the first point you made?",
    "How would I measure if that's working?",
    "What could go wrong with that approach?",
    "Summarize everything we discussed into 3 bullet points"
]

# Print a nice header so we know our automated test is starting
print("\n--- Starting Conversation with PM Advisor ---\n")

# A fake "session ID" to track this specific chat room
config = {"configurable": {"session_id": "pm_chat_01"}}

# 6. Loop through our 5 predefined messages sequentially to simulate a real-time conversation
for message_index, message in enumerate(test_conversation, 1):
    
    print(f"Message {message_index} (You): {message}\n")
    
    # Trigger our memory-wrapped agent! 
    # Under the hood, it injects the history, asks the AI, and saves the new answer back to history automatically!
    response = conversational_agent.invoke({"human_input": message}, config=config)
    
    print(f"Advisor: {response}")
    print("\n" + "="*50 + "\n")

# 7. Finally, let's peek inside the actual text log to see what the AI's "memory" physically looked like
print("\n--- Full Conversation History Log (What the AI 'remembered') ---\n")

# Loop through the raw memory database and print every single saved message
for stored_msg in memory_database.messages:
    # Check if the message was stored as a Human or AI message type
    from langchain_core.messages import HumanMessage
    sender = "Human" if isinstance(stored_msg, HumanMessage) else "AI"
    
    print(f"[{sender}]: {stored_msg.content}\n")
    print("-" * 30 + "\n")
