import os
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment variables without hardcoded paths using find_dotenv()
load_dotenv(find_dotenv())

# =====================================================================
# --- For a Product Manager: What is MCP and What is this Server? ---
# 
# MCP (Model Context Protocol) is an open standard that allows AI models 
# to securely connect to diverse data sources. Think of it like a 
# "universal adapter plug."
# 
# Instead of writing custom integration code for ChatGPT, then another 
# for Claude, and another for a local model, we built this MCP Server.
# It wraps our simple JSON files into standardized "Tools". Now, any 
# AI agent that understands MCP can instantly plug into this server 
# and query our customer feedback data securely and efficiently.
# =====================================================================

# Initialize our MCP Server using the FastMCP pattern
mcp = FastMCP("Feedback Intelligence Server")

# Establish relative file paths based on the project root.
# Using Path(__file__).parent.parent safely targets 'projects/02-feedback-loop-agent'
# without relying on brittle hardcoded absolute paths (like C:/Users/...).
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "mock"

def _load_json_file(filename: str):
    """
    Helper function to safely load a JSON file.
    Gracefully handles 'File Not Found' and 'Invalid JSON' errors.
    """
    filepath = DATA_DIR / filename
    
    if not filepath.exists():
        # Returning a clear error dictionary gracefully instead of crashing the app
        return {"error": f"File not found: {filename}. Please ensure the data/mock/ folder contains this file."}
    
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        return {"error": f"Error parsing JSON in file {filename}. Details: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error reading {filename}: {str(e)}"}

def _filter_by_date(data, since_date_str: str):
    """
    Helper function to filter data based on the 'timestamp' field.
    Gracefully handles invalid dates and returns empty lists if nothing matches.
    """
    # If our data load returned an error dictionary, pass it straight through
    if isinstance(data, dict) and "error" in data:
        return data 
        
    try:
        # Convert the user input "YYYY-MM-DD" into a datetime object for comparison
        target_date = datetime.strptime(since_date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": f"Invalid date format: '{since_date_str}'. Please use exactly 'YYYY-MM-DD'."}
        
    filtered_results = []
    
    for item in data:
        raw_timestamp = item.get("timestamp", "")
        try:
            # Parse the timestamp string from the JSON (assumes 'YYYY-MM-DDThh:mm:ssZ' format)
            date_only_string = raw_timestamp.split("T")[0]
            item_date = datetime.strptime(date_only_string, "%Y-%m-%d")
            
            # Keep the item if it occurred on or after the requested date
            if item_date >= target_date:
                filtered_results.append(item)
        except ValueError:
            # If an individual item has a bad timestamp, we skip it gracefully
            continue
            
    # As requested, if no matches are found, we return an empty list, not an error.
    return filtered_results

# ---------------------------------------------------------------------
# EXPOSED AI TOOLS
# The @mcp.tool() decorator automatically exposes these python functions 
# to connecting AI agents as invokable capabilities.
# ---------------------------------------------------------------------

@mcp.tool()
def get_appstore_reviews(since_date: str):
    """
    Returns all App Store reviews submitted after the given date.
    Date must be strictly in "YYYY-MM-DD" format.
    Returns full individual JSON items including user/product metadata.
    """
    data = _load_json_file("appstore_reviews.json")
    return _filter_by_date(data, since_date)

@mcp.tool()
def get_nps_responses(since_date: str):
    """
    Returns all Net Promoter Score (NPS) surveys submitted after the given date.
    Date must be strictly in "YYYY-MM-DD" format.
    Returns full individual JSON items including user/product metadata.
    """
    data = _load_json_file("nps_surveys.json")
    return _filter_by_date(data, since_date)

@mcp.tool()
def get_sales_notes(since_date: str):
    """
    Returns all enterprise Sales Call Notes submitted after the given date.
    Date must be strictly in "YYYY-MM-DD" format.
    Returns full individual JSON items including sales strategy metadata.
    """
    data = _load_json_file("sales_call_notes.json")
    return _filter_by_date(data, since_date)

@mcp.tool()
def get_all_feedback(since_date: str):
    """
    Combines and returns feedback from all three sources (App Store, NPS, Sales) 
    that occurred after the given date (format: "YYYY-MM-DD").
    
    Each returned item is enriched with a 'source' key so the core agent 
    understands origin tracking.
    """
    # Fetch data using our existing secure functions
    appstore = get_appstore_reviews(since_date)
    nps = get_nps_responses(since_date)
    sales = get_sales_notes(since_date)

    combined_results = []
    
    # We append a 'source' tag manually so the AI can distinguish the records
    if isinstance(appstore, list):
        for item in appstore:
            item["source"] = "app_store_review"
            combined_results.append(item)
            
    if isinstance(nps, list):
        for item in nps:
            item["source"] = "nps_survey" 
            combined_results.append(item)
            
    if isinstance(sales, list):
        for item in sales:
            item["source"] = "sales_call"
            combined_results.append(item)
            
    return combined_results

def _get_single_source_stats(filename: str, readable_name: str):
    """Internal helper to calculate summary stats for a single data file."""
    data = _load_json_file(filename)
    
    # Check if a gracefull error was cast (like file not found)
    if isinstance(data, dict) and "error" in data:
        return {"name": readable_name, "total_items": 0, "date_range": "Not Available", "error": data["error"]}
        
    if not data:
        return {"name": readable_name, "total_items": 0, "date_range": "No Data"}
        
    dates = []
    for item in data:
        try:
            item_date = datetime.strptime(item.get("timestamp", "").split("T")[0], "%Y-%m-%d")
            dates.append(item_date)
        except ValueError:
            pass
            
    if not dates:
        return {"name": readable_name, "total_items": len(data), "date_range": "Unknown"}
        
    # Format the oldest to newest date found in the dataset
    min_date = min(dates).strftime("%Y-%m-%d")
    max_date = max(dates).strftime("%Y-%m-%d")
    return {"name": readable_name, "total_items": len(data), "date_range": f"{min_date} to {max_date}"}

@mcp.tool()
def get_sources():
    """
    Returns an overview of all available feedback data sources.
    Includes the human readable name, total item count, and the date span available.
    """
    return [
        _get_single_source_stats("appstore_reviews.json", "App Store Reviews"),
        _get_single_source_stats("nps_surveys.json", "NPS Surveys"),
        _get_single_source_stats("sales_call_notes.json", "Sales Call Notes")
    ]

# =====================================================================
# SERVER EXECUTION
# This is where the magic happens. When another script or AI agent 
# executes this file, it will call `mcp.run()`. 
# 
# CRITICAL FOR MCP: This server communicates with other agents using 
# standard input and output (stdio). Because of this, we MUST NOT use 
# regular `print()` statements anywhere in this file! Doing so would 
# corrupt the JSON communication protocol used by MCP.
# =====================================================================
if __name__ == "__main__":
    # Start the FastMCP server to listen for incoming connections 
    # via the standard MCP protocol.
    mcp.run()
