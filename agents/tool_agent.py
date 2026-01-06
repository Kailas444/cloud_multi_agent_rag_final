from tools.weather_tool import weather_tool_call
from tools.calculator_tool import calculator_tool_call

def tool_execution_agent(state):
    state["react_steps"].append({"act":"Tool.call","tool":state["tool_name"]})
    if state["tool_name"]=="weather":
        state["tool_result"]=weather_tool_call(state["tool_input"])
    elif state["tool_name"]=="calculator":
        state["tool_result"]=calculator_tool_call(state["tool_input"])
    state["react_steps"].append({"observe":state["tool_result"]})
    return state
