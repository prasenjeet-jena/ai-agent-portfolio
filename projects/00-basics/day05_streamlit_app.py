import streamlit as st
import sys
import os

# Add the current directory to path so we can import directly from our Day 04 agent!
# This ensures it works no matter where you run the command from.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 8. Importing the exact AI workflow we built previously
from day04_langgraph_intro import workflow

# ======================================================================
# PM CONCEPT: Streamlit (Rapid Prototyping)
# ======================================================================
# Streamlit is a magical tool that turns Python scripts into real web apps instantly!
# It's perfect for PMs because you can build internal AI tools without learning 
# complex HTML/CSS or begging your frontend team for resources.

# 1. Page Configuration
# We set this first to make sure the app uses a clean, full-width layout.
st.set_page_config(
    page_title="Product Feedback Intelligence",
    page_icon="🧠",
    layout="centered"
)

# 10. Professional Custom Styling (Premium Feel)
# We inject a tiny bit of CSS to make our app look premium, avoiding the default grey look.
st.markdown("""
<style>
    /* Clean styling for our categorized badges */
    .badge-bug {
        background-color: #ff4b4b; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.9em;
    }
    .badge-feature {
        background-color: #1f77b4; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.9em;
    }
    .badge-general {
        background-color: #2ca02c; color: white; padding: 6px 14px; border-radius: 20px; font-weight: bold; font-size: 0.9em;
    }
    
    /* Styling for the AI response box to make it look professional and clean */
    .response-box {
        background-color: #ffffff; 
        padding: 24px; 
        border-radius: 12px; 
        border-left: 6px solid #4a4a4a; 
        box-shadow: 0 4px 12px rgba(0,0,0,0.08); 
        margin-top: 10px;
        color: #1a1a1a;
        font-family: inherit;
    }
    /* Force black text on white background so it's always readable */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 8px !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Give buttons a nice minimum height and consistent styling */
    .stButton>button {
        border-radius: 8px !important;
        font-weight: bold !important;
        min-height: 50px !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stButton>button:hover {
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Professional Header
st.title("🧠 Product Feedback Intelligence")
st.subheader("AI-powered feedback classification and response system")
st.markdown("---")

# ======================================================================
# 6. PM CONCEPT: State Management (Short-term Memory)
# ======================================================================
# Streamlit forgets EVERYTHING when a user clicks a button (it re-runs the whole code top to bottom).
# We use 'session_state' like a temporary memory vault. If we don't save the AI's result here, 
# it will vanish the moment the screen refreshes!
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "history" not in st.session_state:
    st.session_state.history = []

# 2. Text Area Input
st.markdown("**User Feedback**")
user_input = st.text_area(
    "User Feedback", 
    placeholder="Paste your user feedback here...", 
    height=150,
    label_visibility="collapsed"  # Hides the default small label for a cleaner look
)

# Button Layout (Spacing)
# We use columns so the two buttons sit uniformly side-by-side
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    # 3. Analyze Button
    analyze_btn = st.button("Analyze Feedback", type="primary", use_container_width=True)
with col2:
    # 7. Clear Button
    clear_btn = st.button("Clear Data", type="secondary", use_container_width=True)

# 7. If "Clear Data" is clicked, reset memory and instantly redraw the screen!
if clear_btn:
    st.session_state.analysis_result = None
    st.session_state.history = []
    st.rerun()

# 3. Logic when "Analyze" is clicked
if analyze_btn:
    if not user_input.strip():
        st.warning("Whoops! Please paste some feedback before analyzing.")
    else:
        # 4. Show a spinner with the exact required message while the AI is computing
        with st.spinner("Analyzing feedback..."):
            try:
                # Prepare the 'clipboard' exactly like we did in the terminal script
                initial_state = {
                    "user_input": user_input,
                    "category": "",
                    "response": "",
                    "requires_human": False,
                    "handled_by": ""
                }
                
                # Hand the clipboard to our LangGraph Manager (Invoking the multi-agent workflow)
                result = workflow.invoke(initial_state)
                
                # 6. Save the computed result securely into our short-term memory vault
                st.session_state.analysis_result = result
                
                # Also save a quick summary to our Audit History
                new_entry = {
                    "Input": user_input,
                    "Category": result.get("category"),
                    "Requires Human": "Yes" if result.get("requires_human") else "No",
                    "Handled By": result.get("handled_by")
                }
                st.session_state.history.insert(0, new_entry) # Put freshest at the top
                st.session_state.history = st.session_state.history[:5] # Keep only the last 5
                
            except Exception as e:
                # 9. Polite Error Handling
                # If the AI breaks (e.g., missing API keys), show a beautifully formatted error message.
                st.error("⚠️ The AI agent encountered an issue logging this feedback. Please check your API keys.")
                with st.expander("Show Technical Details"):
                    st.error(f"Error: {str(e)}")

# ======================================================================
# 5. DISPLAYING THE RESULTS 
# ======================================================================
if st.session_state.analysis_result:
    result = st.session_state.analysis_result
    category = result.get('category', 'general')
    
    st.markdown("### 📊 Agent Analysis")
    st.write("") # Whitespace
    
    # Render the correct colored badge based on the category
    if category == "bug_report":
        st.markdown('<span class="badge-bug">🐞 Bug Report</span>', unsafe_allow_html=True)
    elif category == "feature_request":
        st.markdown('<span class="badge-feature">✨ Feature Request</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-general">📬 General</span>', unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 5. Warning Box (If human intervention is technically mandated by the agent flag)
    if result.get('requires_human'):
        st.warning("⚠️ **Requires Human Review:** This ticket has been escalated. An engineer or PM must review it.")
    else:
        st.success("✅ **Fully Handled:** No further human review required.")
        
    # 5. Clean, Professional Box for the AI Response
    st.markdown("**Auto-Drafted Client Response:**")
    st.markdown(f'<div class="response-box">{result.get("response")}</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # 5. Audit Trail: Transparently show which AI specialized agent managed the request
    st.caption(f"🤖 **Audit Trail:** Routed to and fully handled by `{result.get('handled_by')}`")

# ======================================================================
# PM CONCEPT: Feedback History (Audit Log)
# ======================================================================
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Recent Analysis History (Last 5)")
    st.dataframe(
        st.session_state.history,
        use_container_width=True,
        hide_index=True
    )
