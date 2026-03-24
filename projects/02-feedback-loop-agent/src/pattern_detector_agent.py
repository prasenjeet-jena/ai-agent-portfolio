import os
import sys
import json
import asyncio
from typing import List, Literal
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

# Ensure environment variables are loaded securely
load_dotenv(find_dotenv())

# =====================================================================
# --- For a Product Manager: What is this Pattern Detector Agent? ---
# 
# While your previously built "Monitor Agent" looks at feedback ONE by 
# ONE (Micro view), this Pattern Detector Agent looks at the MACRO picture. 
# It takes entire batches of enriched feedback and performs 
# "Zero-Shot Clustering."
# 
# This means the AI dynamically groups similar problems together into 
# broader Feature Themes (like "UI/UX" or "Performance") without us 
# having to pre-program or guess those buckets beforehand. 
# 
# It highlights emerging risks and immediately yields tactical 
# recommendations on what your engineering team should prioritize next.
# =====================================================================

class Cluster(BaseModel):
    theme_name: str = Field(description="The primary product theme, e.g., 'Performance', 'UI/UX', 'Pricing', 'Exporting'")
    count: int = Field(description="The approximate number of items belonging to this theme")
    summary_of_issues: str = Field(description="A concise summary of the core problems described in this cluster")
    priority_level: Literal["High", "Medium", "Low"] = Field(description="Priority based on urgency, frequency, and business impact")
    feedback_ids: List[str] = Field(description="List of feedback_ids (e.g., 'appstore_1') belonging to this cluster")

class SourceCount(BaseModel):
    source_name: str = Field(description="The name of the source, e.g., 'App Store', 'NPS Surveys', 'Sales Calls'")
    count: int = Field(description="Number of items from this source belonging to the theme")

class CrossSourcePattern(BaseModel):
    theme_name: str = Field(description="The name of the theme found across multiple sources")
    source_counts: List[SourceCount] = Field(description="List of sources and their respective counts for this theme")
    total_count: int = Field(description="Total number of items in this cross-source pattern")
    feedback_ids: List[str] = Field(description="List of feedback_ids (e.g., 'nps_1', 'sales_1') that comprise this cross-source pattern")

class TrendReport(BaseModel):
    clusters: List[Cluster] = Field(description="Grouped themes of feedback")
    cross_source_patterns: List[CrossSourcePattern] = Field(description="Themes that appear in multiple sources simultaneously")
    emerging_risks: List[str] = Field(description="Brand-new severe complaints or escalations that appear highly critical or novel")
    pm_recommendation: str = Field(description="Exactly a 2-sentence tactical advice for what the Product Manager should prioritize")


async def detect_patterns(enriched_data: List[dict]) -> TrendReport:
    """
    Takes a batch of enriched feedback items (output from Monitor Agent) 
    and uses GPT-4o-mini structured outputs to extract macro trends.
    """
    if not enriched_data:
        raise ValueError("No enriched data provided for pattern detection.")

    # Compress the payload: The LLM doesn't need all the redundant metadata 
    # to understand the core themes. By condensing this, we save tokens, process faster,
    # and guarantee the LLM fits 100+ items inside its context window effortlessly.
    condensed_payload = []
    
    for item in enriched_data:
        enrichment = item.get("enrichment", {})
        intent = enrichment.get("intent", "unknown")
        
        # Heuristic filtering: Skip pure praise to maximize the Signal-to-Noise ratio
        if intent in ["praise"]:
            continue
            
        text = item.get("raw_content", {}).get("text", "")
        condensed_payload.append({
            "id": item.get("feedback_id", "unknown"),
            "source": item.get("source", "unknown"),
            "intent": intent,
            "text": text
        })

    # Spin up the Async OpenAI Client
    client = AsyncOpenAI()

    prompt = f"""
You are an expert Lead Product Manager synthesizing raw user feedback data.

Please perform zero-shot clustering on the following batch of user feedback. 
Your goal is to extract clear "Product Themes" (e.g., Performance, UI/UX, Pricing, Data Integration) from the dataset.

CRITICAL INSTRUCTIONS:
1. Signal over Noise: Ignore generic feedback. Focus strictly on actionable signals: `complaint`, `bug_report`, `feature_request`, `escalation`, and `praise` (to capture what's working well).
2. Group the feedback logically into clusters based on the core issue or theme. For EACH cluster, you MUST include the `feedback_ids` of the items you grouped there.
3. Identify cross-source patterns where the same basic theme appears in multiple distinct sources (e.g., App Store + NPS). Populate `cross_source_patterns` with the theme, source counts, and the specific `feedback_ids` of the relevant items.
4. Identify any 'emerging_risks' — these are severe complaints or churn threats that seem highly impactful.
5. Output exactly 2 sentences of tactical `pm_recommendation` detailing the immediate next steps for the engineering/product team (balancing fixes with doubling down on what users love).

Data Batch:
{json.dumps(condensed_payload, indent=2)}
"""

    try:
        # Use OpenAI's native Structured Outputs for flawlessly parsing our TrendReport schema
        response = await client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a strategic AI Product Management assistant."},
                {"role": "user", "content": prompt}
            ],
            response_format=TrendReport,
            temperature=0  # Keeping this at 0 ensures analytical, non-hallucinated classifications
        )
        return response.choices[0].message.parsed
        
    except Exception as e:
        print(f" [!] Error generating pattern detection: {e}")
        raise e

# =====================================================================
# INTERNAL TEST SUITE
# =====================================================================
async def test_run():
    # If this file is run directly, we will grab the actual output from 
    # our previous Monitor Agent and pipe it natively into this new Agent!
    print("\n" + "="*65)
    print(" Feedback Intelligence Agent - Pattern Detector Test Suite")
    print("="*65 + "\n")
    
    # Dynamically import the Monitor Agent from within the same folder seamlessly
    try:
        from monitor_agents import FeedbackMonitor
    except ImportError:
        print(" [!] Make sure you run this script from inside the src/ folder or project root.")
        sys.exit(1)
        
    monitor = FeedbackMonitor()
    
    print(" [*] Connecting to MCP to gather live data...")
    await monitor.connect_to_mcp()
    
    try:
        print(" [*] Running Monitor Agent on All Sources (fetching & enriching)...")
        
        # 1. Fetch from App Store
        appstore_data = await monitor.monitor_appstore("2025-01-01")
        for item in appstore_data: item["source"] = "App Store"
        print(f" [+] Fetched {len(appstore_data)} App Store reviews.")
        
        # 2. Fetch from NPS
        nps_data = await monitor.monitor_nps("2025-01-01")
        for item in nps_data: item["source"] = "NPS Surveys"
        print(f" [+] Fetched {len(nps_data)} NPS surveys.")
        
        # 3. Fetch from Sales Calls
        sales_data = await monitor.monitor_sales("2025-01-01")
        for item in sales_data: item["source"] = "Sales Calls"
        print(f" [+] Fetched {len(sales_data)} Sales call notes.")
        
        # Combine all items into one comprehensive dataset
        enriched_data = appstore_data + nps_data + sales_data
        
        print(f" [+] Success! Combined a total of {len(enriched_data)} enriched feedback items for Pattern Detection.")
        print(" [*] Piping all 100 items into the Pattern Detector Agent (Macro View)...")
        
        # Fire the output into our novel detector agent!
        report = await detect_patterns(enriched_data)
        
        print("\n" + "-"*65)
        print(" 📊 MACRO TREND REPORT")
        print("-"*65)
        
        print(f"\n💡 PM Recommendation:\n{report.pm_recommendation}\n")
        
        print(f"🔥 Emerging Risks ({len(report.emerging_risks)}):")
        for risk in report.emerging_risks:
            print(f"   - {risk}")
            
        print(f"\n📦 Identified Clusters ({len(report.clusters)}):")
        for c in report.clusters:
            print(f"   ► [{c.priority_level} PRIORITY] {c.theme_name} ({c.count} items)")
            print(f"      Summary: {c.summary_of_issues}\n")

        print(f"\n🔗 Cross-Source Patterns ({len(report.cross_source_patterns)}):")
        for pattern in report.cross_source_patterns:
            print(f"   {pattern.theme_name} detected in:")
            source_lines = [f"      {sc.source_name} ({sc.count} items)" for sc in pattern.source_counts]
            print(" + \n".join(source_lines))
            print(f"      = CONFIRMED CROSS-SOURCE PATTERN ({pattern.total_count} items total)\n")
            
    finally:
        await monitor.close()
        print(" [+] Teardown complete. Exiting cleanly.")

if __name__ == "__main__":
    asyncio.run(test_run())
