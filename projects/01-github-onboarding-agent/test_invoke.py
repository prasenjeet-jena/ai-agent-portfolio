import sys
import os
sys.path.append(os.path.abspath('.'))
from rag_chain import workflow, app
print("Workflow type:", type(workflow))
print("App type:", type(app))
try:
    workflow.invoke({"original_question": "test"})
except Exception as e:
    print("workflow error:", type(e).__name__, e)

try:
    print(app.invoke({"original_question": "test"}))
except Exception as e:
    print("app error:", type(e).__name__, e)
