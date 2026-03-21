import streamlit as st
import os
import time
import json
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

# ─────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────
st.set_page_config(
    page_title="GitHub Docs Search",
    page_icon="📘",
    layout="wide"
)

# ─────────────────────────────
# ENVIRONMENT VARIABLES
# ─────────────────────────────
# Load all API keys from .env without hardcoding paths
load_dotenv()

# Ensure we import the workflow from our rag_chain module
try:
    from rag_chain import app as workflow
    from langchain_openai import ChatOpenAI
except ImportError as e:
    st.error(f"Missing required imports: {e}")
    st.stop()


# ─────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'current_results' not in st.session_state:
    st.session_state.current_results = None
if 'cache' not in st.session_state:
    st.session_state.cache = {}
if 'total_searches' not in st.session_state:
    st.session_state.total_searches = 0
if 'cache_hits' not in st.session_state:
    st.session_state.cache_hits = 0
if 'search_query' not in st.session_state:
    st.session_state.search_query = ""
if 'is_searching' not in st.session_state:
    st.session_state.is_searching = False
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# ─────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────
def extract_title_from_url(url: str) -> str:
    """Converts a GitHub docs URL path into a readable title."""
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return "GitHub Documentation"
        
        # Get the last segment of the path
        segments = path.split("/")
        last_segment = segments[-1]
        
        # Replace dashes with spaces and title case
        title = last_segment.replace("-", " ").title()
        return title
    except Exception:
        return "GitHub Documentation"

def format_answer_html(text: str) -> str:
    """Converts simple markdown (like bolding and newlines) to HTML for clean display."""
    html_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    html_text = html_text.replace('\n', '<br>')
    return html_text

def generate_related_questions(question: str) -> list:
    """
    Generate 3 related follow-up questions a new GitHub user might ask.
    Returns a list of strings derived from a JSON array response.
    """
    try:
        # Initialise LLM for quick follow-up completions
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        prompt = (
            f"Given this GitHub documentation question: '{question}'\n"
            "Suggest 3 related follow-up questions a new GitHub user might ask.\n"
            "Return EXACTLY a JSON list of 3 strings. Do NOT wrap in markdown code blocks like ```json."
        )
        response = llm.invoke(prompt)
        text = response.content.strip()
        
        # Strip potential markdown formatting
        if text.startswith("```json"):
            text = text[7:]
        if text.endswith("```"):
            text = text[:-3]
            
        questions = json.loads(text.strip())
        if isinstance(questions, list) and len(questions) > 0:
            return questions[:3]
        return []
    except Exception as e:
        print(f"Error generating related questions: {e}")
        return []

def perform_search(query: str):
    """Executes the search via rag_chain, updates session tracking and caches."""
    st.session_state.total_searches += 1
    
    query_key = query.lower().strip()
    
    if query_key in st.session_state.cache:
        st.session_state.cache_hits += 1
        results = st.session_state.cache[query_key]
        results["is_cached"] = True
    else:
        # Run workflow from rag_chain
        initial_state = {"original_question": query}
        results = workflow.invoke(initial_state)
        results["is_cached"] = False
        
        # Generate our related questions
        results["related_questions"] = generate_related_questions(query)
        
        # Store in local cache if we are highly confident
        if results.get("confidence") == "HIGH":
            st.session_state.cache[query_key] = results
            
    # Reset feedback flag for new answers
    st.session_state.feedback_given = False
    st.session_state.current_results = results
    
    # Update search history (keep max 10)
    if not st.session_state.search_history or st.session_state.search_history[0] != query:
        st.session_state.search_history.insert(0, query)
        st.session_state.search_history = st.session_state.search_history[:10]


# ─────────────────────────────
# CUSTOM CSS / BRAND COLORS
# ─────────────────────────────
st.markdown("""
<style>
    /* Global Background and Typography */
    .stApp {
        background-color: #FFFFFF;
        color: #2D2D2D;
    }
    
    /* Header Container */
    .header-container {
        background-color: #1B3A6B;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .header-icon {
        font-size: 2.2rem;
    }
    .header-title-box {
        display: flex;
        flex-direction: column;
    }
    .header-title {
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
        padding: 0;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
        padding: 0;
    }

    /* Primary buttons */
    div.stButton > button[kind="primary"] {
        background-color: #FF6B35;
        color: white;
        border: none;
        font-weight: bold;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #E85A28;
    }

    /* Search Subtitle Text */
    .search-subtitle {
        color: #6B7280;
        text-align: center;
        margin-top: -10px;
        margin-bottom: 2rem;
        font-size: 0.9rem;
    }

    /* Summary Card */
    .summary-card {
        background-color: #F0F4FF;
        border-left: 4px solid #1B3A6B;
        padding: 1.5rem;
        border-radius: 4px;
        border-top: 1px solid #E5E7EB;
        border-right: 1px solid #E5E7EB;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .summary-title {
        font-weight: 600;
        font-size: 1.3rem;
        margin-bottom: 1rem;
        color: #1B3A6B;
    }
    .badges-container {
        position: absolute;
        top: 1.5rem;
        right: 1.5rem;
        display: flex;
        gap: 10px;
    }
    .badge {
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 700;
        color: white;
    }
    .badge-high { background-color: #22C55E; }
    .badge-medium { background-color: #F59E0B; }
    .badge-low { background-color: #EF4444; }
    .badge-cache { background-color: #FF6B35; }

    /* Warning Box for Low Confidence */
    .warning-box {
        background-color: #FEF2F2;
        border: 1px solid #EF4444;
        color: #B91C1C;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1.5rem;
        font-size: 0.95rem;
    }

    /* Source Cards Layout */
    .source-card {
        background-color: #FFFFFF;
        border-left: 3px solid #1B3A6B;
        border-top: 1px solid #E5E7EB;
        border-right: 1px solid #E5E7EB;
        border-bottom: 1px solid #E5E7EB;
        padding: 1.2rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        text-decoration: none;
        color: #2D2D2D;
        display: block;
        transition: background-color 0.2s;
        position: relative;
    }
    .source-card:hover {
        background-color: #F8FAFC;
    }
    .source-title {
        font-weight: 600;
        color: #1B3A6B;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }
    .source-snippet {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 0.8rem;
        line-height: 1.4;
    }
    .source-url {
        font-size: 0.8rem;
        color: #6B7280;
    }
    .source-arrow {
        position: absolute;
        right: 1.5rem;
        top: 50%;
        transform: translateY(-50%);
        color: #FF6B35;
        font-weight: bold;
        font-size: 1.4rem;
    }
    
    /* Headers Fixes */
    h1, h2, h3 {
        color: #1B3A6B;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────
# LAYOUT: SIDEBAR
# ─────────────────────────────
with st.sidebar:
    st.title("Search History")
    if st.session_state.search_history:
        for idx, past_query in enumerate(st.session_state.search_history):
            if st.button(f"🔍 {past_query}", key=f"hist_{idx}", use_container_width=True):
                st.session_state.search_query = past_query
                st.session_state.is_searching = True
                st.rerun()
                
        if st.button("Clear history", use_container_width=True):
            st.session_state.search_history = []
            st.rerun()
    else:
        st.write("No recent searches.")
        
    st.markdown("---")
    
    st.title("Statistics")
    st.write(f"Total searches today: **{st.session_state.total_searches}**")
    st.write(f"Instant answers (cache): **{st.session_state.cache_hits}**")
    hit_rate = 0
    if st.session_state.total_searches > 0:
        hit_rate = (st.session_state.cache_hits / st.session_state.total_searches) * 100
    st.write(f"Hit rate: **{hit_rate:.1f}%**")
    
    st.markdown("---")
    
    # Hardcoded Documentation Coverage as requested
    st.title("Documentation Coverage")
    st.write("Repositories")
    st.progress(0.45, text="45%")
    st.write("Pull Requests")
    st.progress(0.20, text="20%")
    st.write("Actions")
    st.progress(0.15, text="15%")
    st.write("Authentication")
    st.progress(0.10, text="10%")
    st.write("Organizations")
    st.progress(0.10, text="10%")


# ─────────────────────────────
# LAYOUT: MAIN AREA
# ─────────────────────────────

# Header
st.markdown("""
    <div class="header-container">
        <div class="header-icon">🐱</div>
        <div class="header-title-box">
            <div class="header-title">GitHub Documentation Search</div>
            <div class="header-subtitle">Powered by official GitHub documentation</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Search Bar
with st.form(key="search_form", clear_on_submit=False):
    col1, col2 = st.columns([5, 1])
    with col1:
        # If the search just triggered, the query lives in session state
        input_query = st.text_input(
            "Search", 
            value=st.session_state.search_query, 
            placeholder="Search GitHub docs...", 
            label_visibility="collapsed"
        )
    with col2:
        submitted = st.form_submit_button("Search", type="primary", use_container_width=True)
        
if submitted and input_query.strip():
    st.session_state.search_query = input_query.strip()
    st.session_state.is_searching = True
    st.rerun()
    
st.markdown("<div class='search-subtitle'>498 documentation pages indexed</div>", unsafe_allow_html=True)

# Process active query logic
query = st.session_state.search_query

if st.session_state.is_searching:
    # ─────────────────────────────
    # LOADING STATE
    # ─────────────────────────────
    empty_container = st.empty()
    with empty_container.container():
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Searching 498 documentation pages..."):
                time.sleep(0.5)
            with st.spinner("Analysing results..."):
                time.sleep(0.5)
            with st.spinner("Generating summary..."):
                time.sleep(0.5)
    
    try:
        perform_search(query)
    except Exception as e:
        st.session_state.current_results = {"error": str(e)}
        
    st.session_state.is_searching = False
    empty_container.empty()
    st.rerun()

# ─────────────────────────────
# EMPTY STATE
# ─────────────────────────────
elif not st.session_state.current_results and not query:
    st.markdown("### Welcome to GitHub Docs Search!")
    st.write("Try one of these example searches:")
    
    examples = [
        "How do I protect the main branch?",
        "What is a CODEOWNERS file?",
        "How do I set up GitHub Actions?",
        "How do I require PR reviews?"
    ]
    
    # Render example chips using layout
    cols = st.columns(2)
    for idx, ex in enumerate(examples):
        if cols[idx % 2].button(ex, key=f"ex_{idx}", use_container_width=True):
            st.session_state.search_query = ex
            st.session_state.is_searching = True
            st.rerun()

# ─────────────────────────────
# RESULTS AREA
# ─────────────────────────────
elif st.session_state.current_results and query:
    results = st.session_state.current_results
    
    # Error Handing Route
    if "error" in results:
        st.markdown(f"""
        <div class="warning-box">
            <strong>Something went wrong.</strong><br>
            Please try again or visit <a href="https://docs.github.com">docs.github.com</a> directly.<br>
            <small style="opacity:0.75">{results['error']}</small>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        conf = results.get("confidence", "LOW")
        ans = results.get("answer", "")
        chunks = results.get("relevant_chunks", [])
        is_cached = results.get("is_cached", False)
        related = results.get("related_questions", [])
        
        # Scenario: NO RESULTS FOUND (Low confidence and no relevant chunks)
        if conf == "LOW" and len(chunks) == 0:
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background-color: #F9FAFB; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 2rem;">
                <h3 style="color: #4B5563;">No documentation found for this query.</h3>
                <p style="color: #6B7280; font-size: 1.1rem;">Try rephrasing your question or visit <a href="https://docs.github.com" target="_blank" style="color: #1B3A6B; font-weight: 500;">docs.github.com</a> directly.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if related:
                st.markdown("#### Suggested Alternative Searches:")
                cols = st.columns(min(3, len(related)))
                for idx, rq in enumerate(related[:3]):
                    if cols[idx].button(rq, key=f"alt_{idx}", use_container_width=True):
                        st.session_state.search_query = rq
                        st.session_state.is_searching = True
                        st.rerun()

        # Scenario: RESULTS FOUND
        else:
            # SECTION 1 — SUMMARY CARD
            badge_html = ""
            if conf == "HIGH":
                 badge_html += '<span class="badge badge-high">🟢 HIGH CONFIDENCE</span>'
            elif conf == "MEDIUM":
                 badge_html += '<span class="badge badge-medium">🟡 MEDIUM CONFIDENCE</span>'
            else:
                 badge_html += '<span class="badge badge-low">🔴 LOW CONFIDENCE</span>'
                 
            if is_cached:
                 badge_html += '&nbsp;<span class="badge badge-cache">⚡ INSTANT ANSWER</span>'
                 
            warning_html = ""
            if conf == "LOW":
                warning_html = """
                <div class="warning-box">
                    <strong>Limited documentation found.</strong> 
                    Consider visiting <a href="https://support.github.com/" target="_blank">GitHub Support</a> for detailed assistance.
                </div>
                """
                
            ans_html = format_answer_html(ans)
            
            st.markdown(f"""
            <div class="summary-card">
                <div class="badges-container">
                    {badge_html}
                </div>
                <div class="summary-title">Summary</div>
                <div style="font-size: 1.05rem; line-height: 1.6;">
                    {ans_html}
                </div>
                {warning_html}
            </div>
            """, unsafe_allow_html=True)
            
            # SECTION 2 — SOURCE CARDS
            if chunks:
                st.markdown("### 📚 Documentation Sources")
                
                unique_sources = []
                seen_urls = set()
                # Aggregate to display up to 3 distinctive sources
                for chunk in chunks:
                    if chunk['url'] not in seen_urls:
                        seen_urls.add(chunk['url'])
                        unique_sources.append(chunk)
                    if len(unique_sources) == 3:
                        break
                        
                for chunk in unique_sources:
                    url = chunk['url']
                    title = extract_title_from_url(url)
                    # Create snippet preventing huge multi-liners
                    snippet = chunk['text'][:120].replace('\n', ' ') + "..."
                    
                    st.markdown(f"""
                    <a href="{url}" target="_blank" class="source-card">
                        <div class="source-title">📄 {title}</div>
                        <div class="source-snippet">{snippet}</div>
                        <div class="source-url">{url}</div>
                        <div class="source-arrow">→</div>
                        <div style="font-size: 0.85rem; color: #1B3A6B; font-weight: bold; margin-top: 0.5rem;">View documentation &rarr;</div>
                    </a>
                    """, unsafe_allow_html=True)
                    
            # SECTION 3 — RELATED SEARCHES
            if related:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### Related Questions")
                cols = st.columns(len(related[:3]))
                for idx, rq in enumerate(related[:3]):
                    if cols[idx].button(rq, key=f"rel_{idx}", use_container_width=True):
                        st.session_state.search_query = rq
                        st.session_state.is_searching = True
                        st.rerun()
                        
            # SECTION 4 — FEEDBACK ROW
            st.markdown("<br><hr>", unsafe_allow_html=True)
            st.write("**Was this helpful?**")
            
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("👍 Yes", disabled=st.session_state.feedback_given, use_container_width=True):
                    st.session_state.feedback_given = True
                    st.rerun()
            with col2:
                if st.button("👎 No", disabled=st.session_state.feedback_given, use_container_width=True):
                    st.session_state.feedback_given = True
                    st.rerun()
                    
            if st.session_state.feedback_given:
                st.success("Thanks for your feedback!")
                
    st.markdown("<br><br>", unsafe_allow_html=True)
