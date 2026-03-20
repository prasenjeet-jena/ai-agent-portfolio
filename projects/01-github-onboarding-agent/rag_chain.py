import os
from typing import List, Dict, Any, TypedDict
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.graph import StateGraph, END
import chromadb
from pydantic import BaseModel, Field

# ==============================================================================
# Setup & Configuration
# ==============================================================================

# 1. Load the OpenAI API Key from the specified .env file
# We do this so the API key is kept secret and not hardcoded in the script.
env_path = "/Users/swarna/Desktop/ai-agent-portfolio/.env"
load_dotenv(dotenv_path=env_path)

if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in .env file at specified path.")

# 2. Define our AI models
# We use GPT-4o-mini for all our LLM tasks to keep it fast and cost-effective.
llm_rewrite = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_grade = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_generate = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 3. Setup ChromaDB for retrieval
current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "data", "chroma_db")
chroma_client = chromadb.PersistentClient(path=db_path)
collection = chroma_client.get_collection(name="github_docs")

# We use the exact same embedding model that was used to create the database
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")

# ==============================================================================
# Graph State Schema
# ==============================================================================

# This defines the "memory" of our chain.
# As the request passes from node to node, they read and update this state.
class GraphState(TypedDict):
    original_question: str
    rewritten_query: str
    retrieved_chunks: List[Dict[str, Any]]
    relevant_chunks: List[Dict[str, Any]]
    answer: str
    sources: List[str]
    has_sufficient_info: bool

# ==============================================================================
# Node 1: Query Rewriter
# ==============================================================================

def rewrite_query(state: GraphState) -> dict:
    """
    Takes the user's plain English question and rewrites it into technical terms
    optimized for ChromaDB retrieval.
    """
    original_question = state["original_question"]

    # Prompt the LLM to rewrite the query
    system_prompt = (
        "You are a technical query rewriter. Your job is to take a user's plain English question "
        "and rewrite it into an optimized search query using technical terms suitable for retrieving "
        "chunks from a vector database containing GitHub documentation.\n"
        "If the question is a standard greeting (e.g., 'Hi', 'Hello', 'How are you') or entirely unrelated "
        "to GitHub, simply return the question as-is."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Rewrite this question: {original_question}"}
    ]
    
    response = llm_rewrite.invoke(messages)
    rewritten = response.content.strip()
    
    return {"rewritten_query": rewritten}

# ==============================================================================
# Node 2: Retriever
# ==============================================================================

def retrieve_chunks(state: GraphState) -> dict:
    """
    Takes the rewritten query and searches our local ChromaDB for the 5 most
    similar documentation chunks.
    """
    rewritten_query = state.get("rewritten_query", state["original_question"])
    
    # Convert query into an embedding vector
    query_embedding = embeddings_model.embed_query(rewritten_query)
    
    # Query ChromaDB for top 5 closest matches
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )
    
    retrieved_chunks = []
    
    # Unpack the top results from ChromaDB's nested dictionary
    if results["documents"] and len(results["documents"]) > 0:
        docs = results["documents"][0]
        metadatas = results["metadatas"][0]
        
        for doc, metadata in zip(docs, metadatas):
            retrieved_chunks.append({
                "text": doc,
                "url": metadata.get("source", "Unknown Source")
            })
            
    return {"retrieved_chunks": retrieved_chunks}

# ==============================================================================
# Node 3: Relevance Grader
# ==============================================================================

class GraderOutput(BaseModel):
    is_relevant: str = Field(description="Answer YES or NO if the chunk is relevant to the question.")
    reasoning: str = Field(description="One sentence why.")

def grade_relevance(state: GraphState) -> dict:
    """
    Evaluates the 5 retrieved chunks individually to see if they actually contain
    information relevant to the original question. Discards irrelevant chunks.
    """
    original_question = state["original_question"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    
    # Use structured output to force the LLM to return JSON
    llm_structured = llm_grade.with_structured_output(GraderOutput)
    
    relevant_chunks = []
    
    system_prompt = (
        "You are a strict relevance grader. Evaluate the provided documentation chunk to see if it contains "
        "information that can aid in answering the original question. Be strict. "
    )
    
    for chunk in retrieved_chunks:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Original Question: {original_question}\n\nChunk Text:\n{chunk['text']}"}
        ]
        
        try:
            grade = llm_structured.invoke(messages)
            # Only keep the chunk if the LLM graded it YES
            if "YES" in grade.is_relevant.upper():
                relevant_chunks.append(chunk)
        except Exception as e:
            # If parsing fails or rate limits happen, assume not relevant
            pass
            
    has_sufficient_info = len(relevant_chunks) > 0
    
    return {
        "relevant_chunks": relevant_chunks,
        "has_sufficient_info": has_sufficient_info
    }

# ==============================================================================
# Node 4: Answer Generator
# ==============================================================================

def generate_answer(state: GraphState) -> dict:
    """
    Generates the final direct answer based strictly on the relevant chunks.
    Also handles greetings and out-of-scope questions contextually.
    """
    original_question = state["original_question"]
    relevant_chunks = state.get("relevant_chunks", [])
    has_sufficient_info = state.get("has_sufficient_info", False)
    
    context_text = ""
    sources = []
    
    if has_sufficient_info:
        for i, chunk in enumerate(relevant_chunks):
            # Pass the text to the LLM to answer from
            context_text += f"\n--- Chunk {i+1} (Source: {chunk['url']}) ---\n{chunk['text']}\n"
            if chunk['url'] not in sources:
                sources.append(chunk['url'])
                
    # Provided system prompt
    system_prompt = (
        "You are a GitHub documentation assistant.\n"
        "Your primary job is to answer questions based ONLY on the provided context.\n"
        "HOWEVER, you must also handle the following special cases appropriately:\n"
        "1. GREETINGS: If the user says 'Hi' or 'Hello', reply with 'Hello'. If they ask 'How are you', reply with 'I am good and I hope you are doing well'.\n"
        "2. CAPABILITIES: If the user asks how you can help, explain that you can answer questions related to GitHub documentation.\n"
        "3. OUT OF SCOPE: If the question is completely unrelated to GitHub (e.g., 'Who is the president of India' or elections), reply exactly with: 'I don't have the information as I am not connected to web and I am only obliged to answer questions related to GitHub docs'.\n"
        "4. NO CONTEXT: For GitHub-related questions where the provided context does not contain enough information, say: 'I don't have enough information about this.'\n\n"
        "The response can be in mark down format for now with proper formatting.\n"
        "Never speculate beyond the provided context for GitHub questions.\n"
        "Always cite your sources. If answering a greeting or out-of-scope question, you do not need to cite sources."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Question: {original_question}\n\nContext:\n{context_text}"}
    ]
    
    response = llm_generate.invoke(messages)
    answer_text = response.content.strip()
    
    # Format the final response as requested
    final_output = f"ANSWER: {answer_text}\n"
    if sources:
        final_output += f"SOURCES: {', '.join(sources)}"
    else:
        final_output += "SOURCES: None"
        
    return {
        "answer": final_output,
        "sources": sources
    }

# ==============================================================================
# Graph Assembly
# ==============================================================================

# Initialize the state graph with our State schema
workflow = StateGraph(GraphState)

# Add our 4 nodes
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("retrieve", retrieve_chunks)
workflow.add_node("grade", grade_relevance)
workflow.add_node("generate", generate_answer)

# Connect the nodes in a straight sequence
workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", END)

# Compile the graph into an executable application
app = workflow.compile()


# ==============================================================================
# Testing Script
# ==============================================================================

def run_tests():
    test_questions = [
        "What are branch protection rules and how do I set them up?"
    ]
    
    print("\n" + "="*80)
    print("🚀 Running GitHub Onboarding Agent Test Suite")
    print("="*80 + "\n")
    
    for i, question in enumerate(test_questions, 1):
        # We start the graph by passing an initial dictionary matching GraphState keys
        initial_state = {"original_question": question}
        
        # Invoke the graph
        final_state = app.invoke(initial_state)
        
        # Extract the tracked metrics
        rewritten = final_state.get("rewritten_query", "")
        num_retrieved = len(final_state.get("retrieved_chunks", []))
        num_passed = len(final_state.get("relevant_chunks", []))
        answer = final_state.get("answer", "")
        sources = final_state.get("sources", [])
        
        # Print format as requested by the Product Manager
        print(f"----- Question {i}/{len(test_questions)} -----")
        print(f"Original question: {question}")
        print(f"Rewritten query: {rewritten}")
        print(f"Number of chunks retrieved: {num_retrieved}")
        print(f"Number of chunks that passed grading: {num_passed}")
        print(f"Final answer:\n{answer}\n")

if __name__ == "__main__":
    run_tests()
