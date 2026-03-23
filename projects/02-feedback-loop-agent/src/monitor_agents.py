import os
import sys
import json
import asyncio
from typing import Literal, List, Any
from contextlib import AsyncExitStack

from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv, find_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Ensure environment variables are loaded securely without hardcoded paths
load_dotenv(find_dotenv())

# =====================================================================
# --- For a Product Manager: What is this Monitor Agent? ---
# 
# The Monitor Agent acts as our intelligent "Feedback Engine".
# 
# 1. MCP Client: It connects securely to our MCP Server (mcp_server.py) 
#    to retrieve standardized feedback data via a sub-process.
# 2. Enrichment: It takes every piece of raw text (e.g., "Great update, 
#    now my app crashes every 5 seconds!") and sends it to OpenAI (GPT-4o-mini).
# 3. Intent & Sarcasm Engine: Instead of just trusting the raw user rating,
#    the AI reads "between the lines". It can detect sarcasm, figure out 
#    the true intent (whether it's a bug, an escalation, or a hidden feature 
#    request), and structures everything neatly for us.
#
# This file is built to be run directly as a test or imported into a larger app.
# =====================================================================

# Define our robust output structure using Pydantic.
# This forces the LLM to return data exactly in this strictly validated format 
# rather than freeform text, guaranteeing our pipeline won't break upstream.
class FeedbackEnrichment(BaseModel):
    sarcasm_detected: bool = Field(description="True if sarcasm is detected in the text")
    sarcasm_confidence: Literal["high", "medium", "low"]
    true_sentiment: Literal["positive", "negative", "neutral", "mixed"]
    intent: Literal["bug_report", "feature_request", "complaint", "praise", "question", "escalation"]
    intent_confidence: Literal["high", "medium", "low"]
    user_segment_identified: Literal["power_user", "casual_user", "at_risk_user", "new_user"]
    requires_human_review: bool = Field(description="Must be set to True if ANY confidence is 'low'")

class FeedbackMonitor:
    def __init__(self):
        # We use AsyncOpenAI to process multiple items at the same time concurrently.
        self.openai_client = AsyncOpenAI()
        self.mcp_session = None
        # AsyncExitStack properly manages our subprocess lifetime and tears it down safely
        self._exit_stack = AsyncExitStack()

    async def connect_to_mcp(self):
        """
        Connects to our local MCP Server via the standard stdio protocol 
        using the Python MCP SDK.
        """
        # We start the mcp_server.py process similarly to running it in a terminal
        script_path = os.path.join(os.path.dirname(__file__), "mcp_server.py")
        server_params = StdioServerParameters(
            command=sys.executable,  # Uses the current python interpreter securely
            args=[script_path],
            env=None
        )
        
        # Open the communication channels to the standalone server
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        
        # Secure the session and initialize the protocol handshakes
        self.mcp_session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.mcp_session.initialize()

    async def close(self):
        """Safely shuts down the MCP connection."""
        await self._exit_stack.aclose()

    async def enrich_item(self, item: dict) -> dict:
        """
        Takes a single feedback item, feeds the text and surrounding context to GPT-4o-mini, 
        and extracts deeper business insights natively appended to the item.
        """
        raw_text = item.get("raw_content", {}).get("text", "")
        if not raw_text:
            return item

        # Build nuanced context from provided ratings or scores
        score_context = ""
        if "rating" in item.get("raw_content", {}):
            score_context = f"Context: User provided a rating of {item['raw_content']['rating']}/5."
        elif "score" in item.get("raw_content", {}):
            score_context = f"Context: User provided an NPS Score of {item['raw_content']['score']}/10."

        # The System Prompt guiding our AI Agent's intelligence rules
        prompt = f"""
Analyze the following user feedback carefully.

Feedback Text: "{raw_text}"
{score_context}

CRITICAL RULES FOR ENRICHMENT:
1. Sarcasm Detection: Look for positive/enthusiastic words combined with a low rating or complaints about problems (e.g., "Love it when the app crashes"). Look closely for ironic phrasing.
2. Intent Classification:
   - Pay special attention to these patterns which indicate a FEATURE REQUEST not a complaint:
     * 'Why can't I...'
     * 'It would be better if...'
     * 'I wish it had...'
     * 'Please add...'
     * 'Would love to see...'
     * 'Missing feature...'
     * Questions about whether a feature exists
     If any of these patterns appear — classify as feature_request regardless of the overall negative sentiment.
   - Genuine threats of leaving or migrating must be labeled 'escalation'.
   - Determine the true intent beneath the surface words.
3. Human Review Flagging: You must set 'requires_human_review' to True if you are guessing, unsure, or if your sarcasm_confidence or intent_confidence is 'low'.

Ensure deep alignment with the user's actual underlying point. Return the structured JSON exclusively.
"""
        try:
            # We use OpenAI's native Structured Outputs (via parse) to strictly guarantee our schema
            response = await self.openai_client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert product feedback analyst capable of nuanced semantic, intent, and sarcasm detection for enterprise-grade applications."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format=FeedbackEnrichment,
                # Temperature 0 ensures the LLM's classification logic remains absolute, 
                # consistent, and deterministic for reliable enterprise analysis over time.
                temperature=0  
            )
            
            enrichment_result = response.choices[0].message.parsed
            
            # Formally append the results into a new copy of our structure
            enriched_item = item.copy()
            # `model_dump()` natively turns the validated Pydantic object back into a dictionary
            enriched_item["enrichment"] = enrichment_result.model_dump()
            return enriched_item

        except Exception as e:
            print(f" [!] API Error during enrichment: {e}")
            return item

    async def _process_enrichment_batch(self, items: List[dict]) -> List[dict]:
        """Runs enrichment on batches to respect OpenAI rate limits natively."""
        enriched_results = []
        batch_size = 10
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Fan-out: Run all items in this batch concurrently
            tasks = [self.enrich_item(item) for item in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for index, res in enumerate(results):
                if isinstance(res, Exception):
                    print(f" [!] Exception enriching item {batch[index].get('feedback_id')}: {res}")
                    enriched_results.append(batch[index]) # Keep original on absolute failure
                else:
                    enriched_results.append(res)
                    
            # Fan-in & Sleep: Politely limit our API throughput to prevent 429 Rate Limit Errors
            await asyncio.sleep(1)
            
        return enriched_results

    async def _fetch_from_mcp(self, tool_name: str, since_date: str) -> List[dict]:
        """Internal helper to securely handle Tool calling against the MCP server."""
        if not self.mcp_session:
            raise RuntimeError("MCP Connection was not established. Run connect_to_mcp() first.")
            
        try:
            result = await self.mcp_session.call_tool(tool_name, arguments={"since_date": since_date})
            
            # If the server returned an empty array, FastMCP encodes this as an empty content list
            if not getattr(result, "content", None):
                return []
                
            # FastMCP converts Python lists into multiple TextContent items natively.
            # We must iterate and parse all of them back into our final list securely.
            items = []
            for block in result.content:
                parsed = json.loads(block.text)
                
                # If an error was returned as a dict, handle it natively
                if isinstance(parsed, dict) and "error" in parsed:
                    print(f" [!] Server Error in {tool_name}: {parsed['error']}")
                    return []
                
                # If FastMCP wrapped the whole list in one block
                if isinstance(parsed, list):
                    items.extend(parsed)
                else:
                    items.append(parsed)
                    
            return items
        except Exception as e:
            print(f" [!] Error calling MCP tool {tool_name}: {e}")
            return []

    async def monitor_appstore(self, since_date: str) -> List[dict]:
        """Fetches App Store Reviews from MCP and enriches them via GPT-4o-mini."""
        items = await self._fetch_from_mcp("get_appstore_reviews", since_date)
        return await self._process_enrichment_batch(items)

    async def monitor_nps(self, since_date: str) -> List[dict]:
        """Fetches NPS Surveys from MCP and enriches them via GPT-4o-mini."""
        items = await self._fetch_from_mcp("get_nps_responses", since_date)
        return await self._process_enrichment_batch(items)

    async def monitor_sales(self, since_date: str) -> List[dict]:
        """Fetches Sales Notes from MCP and enriches them via GPT-4o-mini."""
        items = await self._fetch_from_mcp("get_sales_notes", since_date)
        return await self._process_enrichment_batch(items)

# =====================================================================
# INTERNAL TEST SUITE
# Only executes when this script is run directly. 
# It validates that both the MCP Server sub-process and OpenAI enrichment 
# are correctly bridged in real time.
# =====================================================================
async def main():
    print("\n" + "="*60)
    print(" Feedback Intelligence Agent - Monitor Test Suite")
    print("="*60 + "\n")
    
    # Assert proper key configurations cleanly.
    if "OPENAI_API_KEY" not in os.environ:
        print(" [!] ERROR: OPENAI_API_KEY is not set in environment or .env file.")
        print("     Make sure you have an API key active to test the OpenAI enrichment.")
        sys.exit(1)

    monitor = FeedbackMonitor()
    print(" [*] Booting up MCP connection...")
    await monitor.connect_to_mcp()
    print(" [+] Successfully connected to MCP Hub.\n")

    test_date = "2025-01-01"

    # Define the 3 routines dynamically to keep code clean.
    monitors_to_test = [
        ("App Store Reviews", monitor.monitor_appstore),
        ("NPS Surveys", monitor.monitor_nps),
        ("Sales Call Notes", monitor.monitor_sales)
    ]

    try:
        for name, monitor_func in monitors_to_test:
            print(f"\n--- Testing {name} Pipeline ---")
            print(f" [*] Fetching and enriching data since {test_date}...")
            
            # Execute pipeline
            enriched_items = await monitor_func(test_date)
            total_items = len(enriched_items)
            
            if total_items == 0:
                print(" [!] No items found. Skipping stats.")
                continue

            # Tabulate robust metrics on the results cleanly
            sarcasm_count = sum(1 for item in enriched_items if item.get("enrichment", {}).get("sarcasm_detected") is True)
            human_review_count = sum(1 for item in enriched_items if item.get("enrichment", {}).get("requires_human_review") is True)
            
            intent_counts = {}
            for item in enriched_items:
                intent = item.get("enrichment", {}).get("intent", "unknown")
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Render Metrics
            print(f" [+] Success! Enriched {total_items} items.")
            print(f"     -> Sarcasm Detected: {sarcasm_count}")
            print(f"     -> Requires Human Review: {human_review_count}")
            print(f"     -> Intent Breakdown: {json.dumps(intent_counts)}")
            
            # Display 3 Sample Data Points securely.
            print("\n     -> Previewing 3 Sample Enriched Items:\n")
            samples = enriched_items[:3]
            for i, item in enumerate(samples):
                text = item.get("raw_content", {}).get("text", "")[:70] + "..." # Truncate for display
                enrichment = item.get("enrichment", {})
                print(f"        Sample {i+1}: \"{text}\"")
                print(f"          - Intent: {enrichment.get('intent')} | Sarcasm: {enrichment.get('sarcasm_detected')} | Human Review: {enrichment.get('requires_human_review')}\n")

    finally:
        # Guarantee teardown of sub-processes so they don't leak memory natively.
        print("\n [*] Closing MCP connections...")
        await monitor.close()
        print(" [+] Teardown complete. Exiting cleanly.")

if __name__ == "__main__":
    asyncio.run(main())
