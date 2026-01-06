
import re
from typing import Dict

def planning_agent(state: Dict):
    q = state["user_query"].strip()
    ql = q.lower()

    state.update({
        "react_steps": [], "error": "", "retrieved_context": "",
        "citations": [], "tool_result": {}, "tool_input": {}, "tool_name": ""
    })

    if "weather" in ql or "temperature" in ql or "forecast" in ql:
        loc = q.split(" in ",1)[1] if " in " in ql else "Chennai"
        state.update({
            "operation":"tool","tool_name":"weather",
            "tool_input":{"location":loc,"days":3},
            "plan":"Call weather tool (Open-Meteo)."
        })

    elif re.search(r"\d+\s*[+\-*/]\s*\d+", q):
        expr = re.sub(r"(?i)\bcalculate\b","",q).strip()
        state.update({
            "operation":"tool","tool_name":"calculator",
            "tool_input":{"expression":expr},
            "plan":"Call calculator tool."
        })
    else:
        state.update({"operation":"rag","plan":"Use RAG for grounded answer."})

    state["react_steps"].append({"reason":state["plan"]})
    return state
