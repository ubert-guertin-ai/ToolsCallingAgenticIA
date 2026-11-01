from langgraph.graph import StateGraph, START, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from typing import NotRequired, TypedDict
from langchain_anthropic import ChatAnthropic

##### GLOBAL ######
product_price = {
    'pain': 4,
    'banane': 2,
    'pomme': 3
}

######## TOOLS #############
@tool
def get_product_price_by_name(product_name: str):
    """Give product price with its name"""
    price = product_price.get(product_name.lower())
    if price is None:
        return f"Le produit '{product_name}' n'a pas été trouvé"
    return f"Le prix du produit '{product_name}' est de '{price}$'"

##############################

llm = ChatAnthropic(model_name="claude-haiku-5-5", timeout=None, stop=None)
llm_tool = llm.bind_tools([get_product_price_by_name])


class State(TypedDict):
    request: str
    msg: NotRequired[str]
    

def extract_product_name(state: State):
    messages = [
        SystemMessage("Si le produit est introuvable, ne réessaie pas, jsute dit-le à l'utilisateur"),
        HumanMessage(state["request"])
    ]
    r = llm_tool.invoke(messages)

    if not r.tool_calls:
        return {"msg": r.content}
    
    messages.append(r)

    for call in r.tool_calls:
        result = get_product_price_by_name.invoke(call['args'])
        messages.append(ToolMessage(content=result, tool_call_id=call['id']))
    final = llm_tool.invoke(messages)
    return { "msg": final.content }


g = StateGraph(State)
g.add_node("extract", extract_product_name)
g.add_edge(START, "extract")
g.add_edge("extract", END)

app = g.compile()

print(
    app.invoke({ "request": "Combien coûte la pomme sur l'étage ?" }).get("msg")
)
