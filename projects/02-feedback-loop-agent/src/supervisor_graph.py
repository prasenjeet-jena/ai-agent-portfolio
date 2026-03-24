import json
from datetime import datetime
from typing import TypedDict, Annotated, List, Any
import operator
from dotenv import load_dotenv, find_dotenv
from langgraph.graph import StateGraph, START, END

# Ensure environment variables are loaded securely without hardcoded paths
load_dotenv(find_dotenv())

# =====================================================================
# --- For a Product Manager: What is AgentState? ---
#
# Think of AgentState as a shared "digital clipboard" that gets passed
# between every agent in our pipeline.
#
# When Agent A (the Monitor) finishes its work, it writes its results
# onto this clipboard. Then Agent B (the Pattern Detector) picks up
# that same clipboard, reads what Agent A wrote, does its own analysis,
# and writes ITS results back onto it too.
#
# This is how our agents seamlessly collaborate without ever needing
# to "talk" to each other directly. They just read from and write to
# the same shared clipboard (AgentState).
#
# --- What does each field on this clipboard hold? ---
#
# 1. raw_feedback:
#    The original, unprocessed feedback items fetched from MCP.
#    Each agent APPENDS its fetched items here (via operator.add),
#    so data from App Store, NPS, and Sales all accumulate together.
#
# 2. enriched_feedback:
#    The AI-enriched versions of the raw items. After the Monitor
#    Agent tags each item with intent, sarcasm detection, and
#    sentiment, the enriched result lands here. Again, items from
#    all sources accumulate together via operator.add.
#
# 3. final_report:
#    The finished output from the Pattern Detector Agent. This is
#    the single macro-level trend report (clusters, cross-source
#    patterns, emerging risks, and PM recommendations) that
#    synthesizes ALL enriched feedback into actionable insights.
# =====================================================================

class AgentState(TypedDict):
    raw_feedback: Annotated[List[dict], operator.add]
    enriched_feedback: Annotated[List[dict], operator.add]
    final_report: dict


# =====================================================================
# AGENT INITIALIZATION
# Think of this as hiring our 'Intern' before the workday starts.
# We create the FeedbackMonitor instance once here, so every node
# in the graph can reuse the same intern rather than hiring a new one
# each time.
# =====================================================================
from monitor_agents import FeedbackMonitor
from pattern_detector_agent import detect_patterns

monitor = FeedbackMonitor()


# =====================================================================
# NODE 1: The Data Fetcher — "The Intern"
#
# Imagine you have an intern whose only job is:
#   1. Walk up to the MCP 'Waiter' (our data server).
#   2. Ask: "Give me all App Store reviews, NPS surveys, and Sales call notes since Jan 1, 2025."
#   3. The Waiter hands back the data, and the Intern enriches
#      each item with AI-powered insights (sarcasm, intent, etc.).
#   4. The Intern then places all those enriched items onto our
#      shared 'digital clipboard' (AgentState) so the next agent
#      in the pipeline can pick them up.
#
# This node acts as our "Aggregator"—it ensures the Analyst sees 
# the 'Big Picture' from App Store, NPS, and Sales Call 
# notes all at once.
# =====================================================================
async def fetch_data_node(state: AgentState):
    """
    Graph Node: Connects to MCP, fetches data from all 3 sources, 
    enriches them via the Monitor Agent, and writes results 
    to the shared AgentState clipboard.
    """
    print("\n [Node 1: Aggregator] Starting up...")

    # Step 1: Connect the Aggregator to the MCP Waiter
    await monitor.connect_to_mcp()
    print(" [Node 1] Connected to MCP Server.")

    try:
        # Step 2: Fetch & enrich data from all three sources
        print(" [Node 1] Fetching & enriching from all sources...")
        
        # 1. App Store
        appstore_data = await monitor.monitor_appstore("2025-01-01")
        for item in appstore_data: item["source"] = "App Store"
        print(f" [+] Fetched {len(appstore_data)} App Store reviews.")
        
        # 2. NPS Surveys
        nps_data = await monitor.monitor_nps("2025-01-01")
        for item in nps_data: item["source"] = "NPS Surveys"
        print(f" [+] Fetched {len(nps_data)} NPS surveys.")
        
        # 3. Sales Call Notes
        sales_data = await monitor.monitor_sales("2025-01-01")
        for item in sales_data: item["source"] = "Sales Calls"
        print(f" [+] Fetched {len(sales_data)} Sales call notes.")

        # Combine items from all sources into one big picture dataset
        all_feedback = appstore_data + nps_data + sales_data
        print(f" [Node 1] Done! Combined {len(all_feedback)} total items.")

        # Step 3: Write results to the shared clipboard by returning a dict.
        return {"raw_feedback": all_feedback}

    finally:
        # Step 4: Teardown — always close the MCP connection so it
        # doesn't stay open and leak resources in the background.
        await monitor.close()
        print(" [Node 1] MCP connection closed cleanly.")




# =====================================================================
# NODE 2: The Strategic Analyst — "The Senior Analyst"
#
# After the Intern (Node 1) has fetched and enriched all the raw
# feedback and placed it on the clipboard, the Senior Analyst picks
# it up and does the high-level thinking:
#
#   1. Read ALL enriched feedback items from the clipboard.
#   2. Run our Pattern Detector AI to find macro-level themes,
#      cross-source patterns, and emerging risks.
#   3. Write the finished strategic report back onto the clipboard
#      so leadership (or the next node) can consume it.
#
# This is the "brains" of the operation — turning noisy data into
# clear, actionable product insights.
# =====================================================================
async def analysis_node(state: AgentState):
    """
    Graph Node: Reads enriched feedback from the clipboard,
    runs the Pattern Detector Agent to synthesize macro trends,
    and writes the final TrendReport back to the clipboard.
    """
    print("\n [Node 2: Strategic Analyst] Starting analysis...")

    # Step 1: Pick up the enriched data that Node 1 placed on the clipboard
    enriched_data = state["raw_feedback"]
    print(f" [Node 2] Reading {len(enriched_data)} items from the clipboard.")

    # Step 2: Run the Pattern Detector to synthesize trends
    print(" [Node 2] Running Pattern Detection (zero-shot clustering)...")
    report = await detect_patterns(enriched_data)
    print(" [Node 2] Analysis complete!")

    # Step 3: Write the final report back to the clipboard.
    # We use model_dump() because 'report' is a Pydantic object, and
    # the clipboard (AgentState) expects a standard Python dictionary.
    return {"final_report": report.model_dump()}


# =====================================================================
# THE STATE GRAPH: Designing the Workflow
#
# Think of this as the "Project Manager" designing the office layout.
# We are defining which nodes (workers) exist and how the clipboard
# (the state) flows from one desk to another.
# =====================================================================

# 1. Initialize the Graph using our shared clipboard (AgentState)
workflow = StateGraph(AgentState)

# 2. Add our Nodes (our "workers") to the floor plan
workflow.add_node("fetcher", fetch_data_node)
workflow.add_node("analyst", analysis_node)

# 3. Define the Edges (the "flow of work")
# We start with the Fetcher, then hand the clipboard to the Analyst,
# and then we're finished.
workflow.add_edge(START, "fetcher")
workflow.add_edge("fetcher", "analyst")
workflow.add_edge("analyst", END)

# 4. Compile the Workflow
# This turns our design into an executable 'App' that we can run.
app = workflow.compile()


# =====================================================================
# PERSISTENCE LAYER: Saving the Intelligence
#
# Once our agents finish their work, we need to save the results
# so our React UI can display them.
#
# Think of this as the "Filing Cabinet"—it takes the final report
# from the clipboard, adds a timestamp, and saves it as a JSON
# file that our dashboard will use tomorrow.
# =====================================================================
def save_report(state: AgentState):
    """
    Persistence: Extracts the final report from the state, 
    adds metadata, and saves it to a structured JSON file.
    """
    import os
    
    report = state.get("final_report", {})
    if not report:
        print(" [!] No report found in state. Skipping save.")
        return

    # Add metadata for auditing & display
    report["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "total_items_processed": len(state.get("raw_feedback", []))
    }

    # Include the full raw feedback as a lookup table for the UI
    report["raw_feedback_lookup"] = state.get("raw_feedback", [])

    # Ensure the data/ directory exists
    save_path = os.path.join(os.path.dirname(__file__), "..", "data", "latest_intelligence.json")
    
    # PM EXPLICIT RULE: Always clear old data before writing. 
    # Python's 'w' mode already does this (truncates to 0 bytes),
    # but we'll add a print statement to be crystal clear.
    print(f" [Persistence] Clearing old report and writing fresh data to {os.path.basename(save_path)}...")
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n ✅ Intelligence Report saved to data/latest_intelligence.json")


# =====================================================================
# ISOLATED TEST: The "Start Button"
#
# Run this file directly to trigger the entire automated pipeline. 
# We use 'ainvoke' to kick off the flow. It's like pressing 
# the 'Start' button on a factory assembly line.
# =====================================================================
if __name__ == "__main__":
    import asyncio

    async def test_workflow():
        # Start with a blank clipboard — no data yet
        initial_state = {
            "raw_feedback": [],
            "enriched_feedback": [],
            "final_report": {}
        }

        print("\n" + "=" * 60)
        print(" 🚀 Kicking off the Automated Intelligence Pipeline...")
        print("=" * 60)

        # ainvoke is the 'Start Button'. It passes our blank clipboard
        # into the graph and returns the final finished state after
        # all nodes have completed their work.
        final_state = await app.ainvoke(initial_state)

        # --- Persistence: Save the finished report for the UI ---
        # This JSON file is the 'Bridge' that our React UI will use 
        # tomorrow to display the dashboard.
        save_report(final_state)

        # Display the results for verification
        print("\n" + "=" * 60)
        print(" --- FINAL PIPELINE RESULTS ---")
        print("=" * 60)

        items = final_state.get("raw_feedback", [])
        print(f"\n ✅ Total items processed: {len(items)}")

        report = final_state.get("final_report", {})
        print(f"\n 💡 PM Recommendation:\n    {report.get('pm_recommendation', 'N/A')}")

        print(f"\n 🔥 Emerging Risks ({len(report.get('emerging_risks', []))}):\n")
        for risk in report.get("emerging_risks", []):
            print(f"    - {risk}")

        print(f"\n 📦 Clusters ({len(report.get('clusters', []))}):\n")
        for c in report.get("clusters", []):
            print(f"    ► [{c['priority_level']} PRIORITY] {c['theme_name']} ({c['count']} items)")
            print(f"      {c['summary_of_issues']}\n")

    asyncio.run(test_workflow())
