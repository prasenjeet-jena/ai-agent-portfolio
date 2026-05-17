import sys
import os
sys.path.append(os.path.abspath('.'))
from rag_chain import app

print("App type:", type(app))

try:
    print("Invoking app...")
    result = app.invoke({"original_question": "What is GitHub?"})
    print("Success! Result keys:", result.keys())
    print("Confidence:", result.get("confidence"))
except Exception as e:
    print("Error invoking app:", type(e).__name__, e)
