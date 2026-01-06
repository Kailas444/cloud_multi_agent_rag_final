from agents.rag_agent import retrieve_and_answer
import json

def synthesis_agent(state):
    if state["operation"]=="rag":
        ans,cit = retrieve_and_answer(state["user_query"])
        state["final_answer"]=ans
        state["citations"]=cit
    else:
        state["final_answer"]=json.dumps(state["tool_result"],indent=2)
        state["citations"]=[]
    return state
