import os
from typing import TypedDict
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# Import LangGraph tools to build our workflow
from langgraph.graph import StateGraph, START, END

# 1. Automatically search for the hidden .env file and securely load our secret API keys
load_dotenv(find_dotenv())

# ======================================================================
# PM CONCEPT: The "State" (Think of this as a shared clipboard)
# ======================================================================
# In a multi-agent system, agents (nodes) need a way to pass information
# to each other. We define a strict "clipboard" format called a TypedDict.
# Every time an agent finishes its job, it updates this clipboard and 
# hands it off to the next agent or rule.

class WorkflowState(TypedDict):
    user_input: str       # The original feedback text from the user
    category: str         # The classified category (bug_report, feature_request, general)
    response: str         # The final crafted response
    requires_human: bool  # A flag to escalate to a human if necessary
    handled_by: str       # Extra field: Tracks which specific node answered the user

# We define a strict Pydantic model for our Classifier to ensure it ONLY 
# returns one of our three exact categories. (Like a dropdown menu!)
class CategoryOutput(BaseModel):
    category: str = Field(description="Must be exactly one of: bug_report, feature_request, general")

# ======================================================================
# PM CONCEPT: The "Nodes" (Think of these as Specialized AI Employees)
# ======================================================================
# Why this pattern matters for multi-agent systems:
# Instead of writing one giant, unmanageable "mega-prompt" that tries to do
# everything (classify, write a bug response, write a feature response, etc.), 
# we create tiny, single-purpose agents. This allows us to upgrade how we 
# handle bugs without accidentally breaking how we handle feature requests.
# It makes the AI system reliable, scalable, and easy to debug.

def classifier_node(state: WorkflowState) -> dict:
    """
    Node 1: The 'Dispatcher'. 
    Reads the user_input from the clipboard and tags it with a category.
    """
    # We use temperature=0 because categorization needs strict precision, not creativity
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(CategoryOutput)
    
    prompt = ChatPromptTemplate.from_template(
        "Categorize this user feedback into exactly one of: bug_report, feature_request, general.\n"
        "CRITICAL DISTINCTION:\n"
        "- Someone REQUESTING a feature that doesn't exist yet -> feature_request\n"
        "- Someone ASKING if a feature already exists -> general\n\n"
        "Feedback: {input}"
    )
    
    chain = prompt | structured_llm
    result = chain.invoke({"input": state["user_input"]})
    
    # We return a dictionary matching our WorkflowState. 
    # LangGraph will take this and update the 'clipboard' category.
    return {"category": result.category}

def bug_node(state: WorkflowState) -> dict:
    """
    Node 2: The 'Bug Support' Agent.
    Writes a highly professional, empathetic response to software defects.
    """
    # Temperature=0.7 because client responses should sound natural and human-like
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are a professional customer support agent. A user reported a critical bug:\n'{input}'\n\n"
        "Write a highly professional acknowledgment. You must include:\n"
        "1. A sincere apology\n"
        "2. An acknowledgment of the urgency level\n"
        "3. An expected timeframe for engineering resolution."
    )
    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})
    
    # Update the clipboard with our response and trigger the escalation flag
    return {
        "response": result.content,
        "requires_human": True, # Bugs need engineering review
        "handled_by": "bug_node"
    }

def feature_node(state: WorkflowState) -> dict:
    """
    Node 3: The 'Product Manager' Feedback Agent.
    Handles feature requests with enthusiasm and sets realistic timelines.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are a Product Manager responding to a user feature request:\n'{input}'\n\n"
        "Write a professional, encouraging acknowledgment. You must include:\n"
        "1. Deep gratitude for their product idea\n"
        "2. A brief note on how the product team evaluates new requests\n"
        "3. A realistic timeline expectation for this process."
    )
    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})
    
    return {
        "response": result.content,
        "requires_human": False, # Features are logged, no emergency
        "handled_by": "feature_node"
    }

def general_node(state: WorkflowState) -> dict:
    """
    Node 4: The 'General Inquiries' Agent.
    Handles all other questions cleanly and cleanly.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant responding to user feedback:\n'{input}'\n\n"
        "Write a helpful, friendly general response."
    )
    chain = prompt | llm
    result = chain.invoke({"input": state["user_input"]})
    
    return {
        "response": result.content,
        "requires_human": False,
        "handled_by": "general_node"
    }


# ======================================================================
# PM CONCEPT: Conditional Routing (The "Traffic Cop")
# ======================================================================
# What is this doing? This uses regular Python code (not AI!) to look at 
# the 'clipboard' category after the classifier agent is done. Based on 
# what it sees, it cleanly routes the clipboard to the appropriate next agent.
def route_by_category(state: WorkflowState) -> str:
    category = state.get("category", "general")
    if category == "bug_report":
        return "bug_node"
    elif category == "feature_request":
        return "feature_node"
    else:
        return "general_node"


# ======================================================================
# BUILDING THE GRAPH (Assembling the Factory Floor)
# ======================================================================

# Start with our clipboard definition
builder = StateGraph(WorkflowState)

# 1. Add all our specialized workers (nodes) to the workflow
builder.add_node("classifier_node", classifier_node)
builder.add_node("bug_node", bug_node)
builder.add_node("feature_node", feature_node)
builder.add_node("general_node", general_node)

# 2. Start the assembly line: From START straight to our Dispatcher
builder.add_edge(START, "classifier_node")

# 3. Add 'Conditional Routing' (Decision branching)
# Once 'classifier_node' finishes, run 'route_by_category'.
# The dictionary matches Python strings to the names of our nodes.
builder.add_conditional_edges(
    "classifier_node",
    route_by_category,
    {
        "bug_node": "bug_node",
        "feature_node": "feature_node",
        "general_node": "general_node"
    }
)

# 4. Wherever the route ends up, ensure the process stops afterward.
# All nodes end at END after responding.
builder.add_edge("bug_node", END)
builder.add_edge("feature_node", END)
builder.add_edge("general_node", END)

# Lock everything into a runnable application
workflow = builder.compile()


# ======================================================================
# TEST THE WORKFLOW
# ======================================================================

if __name__ == "__main__":
    # 6. Test with 3 specific PM-focused inputs
    test_inputs = [
        "The export button crashes the app every time on iOS 16",
        "Please add the ability to export data as Excel files",
        "I love your product but want to know if you have an API"
    ]

    print("\n--- Starting LangGraph Multi-Agent Workflow ---\n")

    for text in test_inputs:
        # Give the system a clean slate (clipboard) for each new request
        initial_state = {
            "user_input": text,
            "category": "",
            "response": "",
            "requires_human": False,
            "handled_by": ""
        }
        
        # Trigger the workflow!
        final_state = workflow.invoke(initial_state)
        
        # 7. Print requested outputs neatly
        print(f"Original input: \"{final_state['user_input']}\"")
        print(f"Classified category: {final_state['category']}")
        print(f"Which node handled it: {final_state['handled_by']}")
        print(f"Requires Human: {final_state['requires_human']}")
        print(f"\nFinal response:\n{final_state['response']}")
        print("\n" + "="*70 + "\n")
