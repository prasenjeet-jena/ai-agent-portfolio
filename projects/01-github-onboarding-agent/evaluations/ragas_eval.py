import os
import sys
import pandas as pd
import ragas
from dotenv import load_dotenv

# Add the parent directory to sys.path so we can import rag_chain
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import the compiled LangGraph workflow
from rag_chain import app

# 1. Load the OpenAI API Key automatically without exposing the system path
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

print(f"\n============================================================")
print(f"RAGAS Version: {ragas.__version__}")
print(f"============================================================")

from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

# =====================================================================
# 2. Golden Questions and Ground Truths
# =====================================================================

data = [
    {
        "question": "What are branch protection rules and how do I set them up?",
        "ground_truth": "Branch protection rules enforce workflows to prevent unauthorized changes. You can set them up by navigating to your repository settings, clicking Branches, and adding a branch protection rule."
    },
    {
        "question": "How do I protect the main branch?",
        "ground_truth": "To protect the main branch, go to repository settings, access the Branches section, and add a branch protection rule with the pattern 'main'."
    },
    {
        "question": "What are branch protection rules?",
        "ground_truth": "Branch protection rules are configurations that prevent unauthorized changes such as force pushes or branch deletion, and enforce requirements like code reviews."
    },
    {
        "question": "How do I require pull request reviews?",
        "ground_truth": "You can require pull request reviews by creating a branch protection rule and enabling the option 'Require pull request reviews before merging'."
    },
    {
        "question": "How do I set up GitHub Actions?",
        "ground_truth": "You can set up GitHub Actions by navigating to the Actions tab in your repository and defining workflows using YAML files stored in the .github/workflows directory."
    },
    {
        "question": "What is a CODEOWNERS file?",
        "ground_truth": "A CODEOWNERS file defines individuals or teams responsible for reviewing code changes in specific files or directories."
    },
    {
        "question": "How do I manage repository permissions?",
        "ground_truth": "Repository permissions are managed in the repository Settings under the Access or Collaborators section, where you can invite people and assign access roles."
    },
    {
        "question": "How do I create an organization?",
        "ground_truth": "You can create an organization by clicking your profile picture, selecting 'Your organizations', and clicking 'New organization' to configure its name and billing plan."
    },
    {
        "question": "What is two factor authentication?",
        "ground_truth": "Two-factor authentication (2FA) is an extra security layer. With 2FA, you provide your password and a secondary form of authentication to log in."
    },
    {
        "question": "How do I merge a pull request?",
        "ground_truth": "You merge a pull request by navigating to the pull request page, reviewing the changes, and clicking the 'Merge pull request' button, then confirming."
    },
    {
        "question": "How do I set up branch protection for enterprise?",
        "ground_truth": "For enterprise, you set up protections by going to repository settings and adding branch protection rules. Administrators can also enforce them across multiple repositories using Rulesets."
    },
    {
        "question": "Who is the president of India?",
        "ground_truth": "I don't have the information as I am not connected to web and I am only obliged to answer questions related to GitHub docs."
    },
    {
        "question": "What do you think, who would in USA elections next time?",
        "ground_truth": "I don't have the information as I am not connected to web and I am only obliged to answer questions related to GitHub docs."
    }
]

# =====================================================================
# 3. Connect to Real Pipeline
# =====================================================================

print("\n🚀 Executing Real Pipeline for Evaluation...\n")

samples = []

# Loop over the 13 questions defined
for idx, item in enumerate(data[:13]):
    question = item["question"]
    ground_truth = item["ground_truth"]
    
    # 1. Run through pipeline
    final_state = app.invoke({"original_question": question})
    
    answer = final_state.get("answer", "")
    confidence = final_state.get("confidence", "N/A")
    relevant_chunks = final_state.get("relevant_chunks", [])
    sources = final_state.get("sources", [])
    
    # Extract text chunks for Ragas
    contexts = [chunk["text"] for chunk in relevant_chunks]
    if not contexts:
         contexts = ["No relevant context found."]
         
    num_words = len(answer.split())
    num_sources = len(sources)
    
    # Print metrics required for PM
    print(f"--- Question {idx+1} ---")
    print(f"Question: {question}")
    print(f"Confidence Level: {confidence}")
    print(f"Answer Length (words): {num_words}")
    print(f"Number of Sources Cited: {num_sources}\n")
    
    # Map to Ragas format
    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
        reference=ground_truth
    )
    samples.append(sample)

eval_dataset = EvaluationDataset(samples)

# =====================================================================
# 4. Run all 4 RAGAS metrics
# =====================================================================

print("🚀 Running RAGAS Evaluation on Pipeline Output...\n")

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
evaluator_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

results = evaluate(
    eval_dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings
)

df_results = results.to_pandas()

scores = {
    "Faithfulness": df_results["faithfulness"].mean(),
    "Answer Relevancy": df_results["answer_relevancy"].mean(),
    "Context Recall": df_results["context_recall"].mean(),
    "Context Precision": df_results["context_precision"].mean(),
}

# =====================================================================
# 5. Print Results Report
# =====================================================================

TARGET_SCORE = 0.80
metrics_passed = 0

print("============================================================")
print(f"{'Metric':<20} | {'Score':<10} | {'Target':<10} | {'Pass/Fail':<10}")
print("-" * 60)

for metric_name, score in scores.items():
    if pd.isna(score):
        score = 0.0  
        
    passed = "PASS" if score >= TARGET_SCORE else "FAIL"
    if passed == "PASS":
        metrics_passed += 1
        
    print(f"{metric_name:<20} | {score:<10.4f} | {TARGET_SCORE:<10.4f} | {passed:<10}")

print("============================================================")
print(f"\n{metrics_passed} out of 4 metrics passed target\n")
