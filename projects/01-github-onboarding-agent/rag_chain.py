import os
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Optional
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
# We reuse instantiated models to reduce memory footprint and initialization overhead.
llm_strict = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_creative = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

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
    confidence: str

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
        "to GitHub, simply return the question as-is.\n"
        "Never include search operators like site:, filetype:, or similar operators.\n"
        "Return only plain search terms."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Rewrite this question: {original_question}"}
    ]
    
    response = llm_strict.invoke(messages)
    rewritten = response.content.strip()
    
    return {"rewritten_query": rewritten}

# ==============================================================================
# Node 2: Retriever
# ==============================================================================

def retrieve_chunks(state: GraphState) -> dict:
    """
    Takes the rewritten query and searches our local ChromaDB for the 10 most
    similar documentation chunks.
    """
    rewritten_query = state.get("rewritten_query", state["original_question"])
    
    # Convert query into an embedding vector
    query_embedding = embeddings_model.embed_query(rewritten_query)
    
    # Query ChromaDB for top 10 closest matches
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10
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

# Initialize the structured LLM globally to avoid recreating it in every node execution
llm_grade_structured = llm_strict.with_structured_output(GraderOutput)

def grade_relevance(state: GraphState) -> dict:
    """
    Evaluates the 10 retrieved chunks individually to see if they actually contain
    information relevant to the original question. Discards irrelevant chunks.
    """
    original_question = state["original_question"]
    retrieved_chunks = state.get("retrieved_chunks", [])
    
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
            grade = llm_grade_structured.invoke(messages)
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
        context_parts = []
        for i, chunk in enumerate(relevant_chunks):
            # Pass the text to the LLM to answer from
            context_parts.append(f"\n--- Chunk {i+1} (Source: {chunk['url']}) ---\n{chunk['text']}\n")
            if chunk['url'] not in sources:
                sources.append(chunk['url'])
        context_text = "".join(context_parts)
                
    # Provided system prompt
    system_prompt = (
        "You are a GitHub documentation assistant.\n"
        "Before answering, identify which specific parts of the provided context support your answer.\n\n"
        "STRICT RULES:\n"
        "- Every sentence in your answer must come directly from the provided context\n"
        "- Do NOT use GitHub knowledge from your training data\n"
        "- Do NOT add steps, details, or examples not explicitly in the context\n"
        "- If context is incomplete — say:\n"
        "  'Based on available documentation: [answer].\n"
        "  For complete information visit: [source]'\n\n"
        "HOWEVER, you must also handle the following special cases appropriately:\n"
        "1. GREETINGS: If the user says 'Hi' or 'Hello', reply with 'Hello'. If they ask 'How are you', reply with 'I am good and I hope you are doing well'.\n"
        "2. CAPABILITIES: If the user asks how you can help, explain that you can answer questions related to GitHub documentation.\n"
        "3. OUT OF SCOPE: If the question is completely unrelated to GitHub, reply exactly with: 'I don't have the information as I am not connected to web and I am only obliged to answer questions related to GitHub docs'.\n"
        "The response can be in mark down format for now with proper formatting.\n"
        "If answering a greeting or out-of-scope question, you do not need to cite sources."
    )
    
    user_prompt = (
        f"Context provided:\n{context_text}\n\n"
        f"Question: {original_question}\n\n"
        "First identify relevant context passages, then write your answer strictly from those."
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm_creative.invoke(messages)
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
# Node 5: Confidence Scorer
# ==============================================================================

def score_confidence(state: GraphState) -> dict:
    """
    Evaluates the generated answer against the retrieved chunks to determine a confidence score.
    
    Why this matters for a Product Manager:
    Confidence scoring acts as a safety net. It tells us how much we should trust the AI's answer.
    If an answer has LOW confidence, we might flag it for human review in our product 
    or show a warning to the user so they know the information might be incomplete.
    """
    answer = state.get("answer", "")
    relevant_chunks = state.get("relevant_chunks", [])
    has_sufficient_info = state.get("has_sufficient_info", False)
    
    # If there wasn't enough context to generate a good answer, confidence is inherently LOW
    if not has_sufficient_info:
        return {"confidence": "LOW"}
        
    context_parts = [f"\n--- Chunk {i+1} ---\n{chunk['text']}\n" for i, chunk in enumerate(relevant_chunks)]
    context_text = "".join(context_parts)
        
    system_prompt = (
        "You are an objective evaluation assistant.\n"
        "Given these source chunks and this answer, rate the confidence as:\n"
        "HIGH — answer directly supported by multiple strong sources\n"
        "MEDIUM — answer partially supported, some inference involved\n"
        "LOW — limited source support, answer may be incomplete\n\n"
        "Return EXACTLY one of: HIGH, MEDIUM, LOW"
    )
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Source Chunks:\n{context_text}\n\nGenerated Answer:\n{answer}"}
    ]
    
    response = llm_strict.invoke(messages)
    confidence = response.content.strip().upper()
    
    # Fallback to ensure we always return one of the three options
    if confidence not in ["HIGH", "MEDIUM", "LOW"]:
        confidence = "LOW"
        
    return {"confidence": confidence}

# ==============================================================================
# Graph Assembly
# ==============================================================================

# Initialize the state graph with our State schema
workflow = StateGraph(GraphState)

# Add our 5 nodes
workflow.add_node("rewrite", rewrite_query)
workflow.add_node("retrieve", retrieve_chunks)
workflow.add_node("grade", grade_relevance)
workflow.add_node("generate", generate_answer)
workflow.add_node("score", score_confidence)

# Connect the nodes in a straight sequence
workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "grade")
workflow.add_edge("grade", "generate")
workflow.add_edge("generate", "score")
workflow.add_edge("score", END)

# Compile the graph into an executable application
app = workflow.compile()


# ==============================================================================
# Answer Cache
# ==============================================================================
# Why this matters for a Product Manager:
# Caching saves both time and money. If a user asks a question we've already answered 
# accurately (HIGH confidence), we can serve the stored answer instantly instead of 
# running the expensive AI chain again. This improves user experience through faster 
# response times and reduces our overall API usage costs.

_answer_cache = {}

def get_cached_answer(question: str) -> Optional[dict]:
    """Retrieves an answer from the cache if it exists."""
    key = question.lower().strip()
    if key in _answer_cache:
        _answer_cache[key]["hit_count"] += 1
        return _answer_cache[key]
    return None

def add_to_cache(question: str, final_state: dict):
    """Adds a HIGH confidence answer to the cache."""
    key = question.lower().strip()
    _answer_cache[key] = {
        "answer": final_state.get("answer", ""),
        "sources": final_state.get("sources", []),
        "confidence": final_state.get("confidence", "HIGH"),
        "timestamp": datetime.now().isoformat(),
        "hit_count": 0
    }

def clear_cache():
    """Empties the entire answer cache."""
    _answer_cache.clear()


# ==============================================================================
# Testing Script
# ==============================================================================

def run_tests():
    test_questions = [
        "What are branch protection rules and how do I set them up?",
        "How do I protect the main branch?",
        "What are branch protection rules?",
        "How do I require pull request reviews?",
        "How do I set up GitHub Actions?",
        "What is a CODEOWNERS file?",
        "How do I manage repository permissions?",
        "How do I create an organization?",
        "What is two factor authentication?",
        "How do I merge a pull request?",
        "How do I set up branch protection for enterprise?",
        "Who is the president of India?",
        "What do you think, who would in USA elections next time?",
        "How can i share my repo with my colleague?",
        "How to check if the repository is meeting all the security standards?",
        "How are you?",
        "How do I protect the main branch?"
    ]
    
    print("\n" + "="*80)
    print("🚀 Running GitHub Onboarding Agent Test Suite")
    print("="*80 + "\n")
    
    cache_hits = 0
    cache_misses = 0
    
    for i, question in enumerate(test_questions, 1):
        print(f"----- Question {i}/{len(test_questions)} -----")
        print(f"Original question: {question}")
        
        # 1. Check the cache before running the pipeline
        cached_result = get_cached_answer(question)
        
        if cached_result:
            cache_hits += 1
            print("⚡ Cache hit! Serving verified answer")
            print(f"Rewritten query: N/A (Cached)")
            print(f"Number of chunks retrieved: N/A (Cached)")
            print(f"Number of chunks that passed grading: N/A (Cached)")
            print(f"Confidence level: {cached_result['confidence']}")
            print(f"Final answer:\n{cached_result['answer']}\n")
            continue
            
        cache_misses += 1
        
        # 2. If no cache hit, run the full Graph
        initial_state = {"original_question": question}
        final_state = app.invoke(initial_state)
        
        rewritten = final_state.get("rewritten_query", "")
        num_retrieved = len(final_state.get("retrieved_chunks", []))
        num_passed = len(final_state.get("relevant_chunks", []))
        confidence = final_state.get("confidence", "N/A")
        answer = final_state.get("answer", "")
        
        # 3. Save to cache if we got a HIGH confidence response
        if confidence == "HIGH":
            add_to_cache(question, final_state)
        
        print(f"Rewritten query: {rewritten}")
        print(f"Number of chunks retrieved: {num_retrieved}")
        print(f"Number of chunks that passed grading: {num_passed}")
        print(f"Confidence level: {confidence}")
        print(f"Final answer:\n{answer}\n")
        
    # Print cache statistics
    total_q = cache_hits + cache_misses
    hit_rate = (cache_hits / total_q) * 100 if total_q > 0 else 0
    
    print("="*80)
    print("📊 Cache Statistics Summary")
    print("="*80)
    print(f"Total questions asked: {total_q}")
    print(f"Cache hits: {cache_hits}")
    print(f"Cache misses: {cache_misses}")
    print(f"Cache hit rate: {hit_rate:.1f}%\n")

if __name__ == "__main__":
    run_tests()
