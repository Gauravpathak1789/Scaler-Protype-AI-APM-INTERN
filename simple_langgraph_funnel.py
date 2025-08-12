# libraries
from dotenv import load_dotenv
import os, sqlite3
from typing import TypedDict, Annotated
from email.message import EmailMessage
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

# Config
llm= ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.2)

# State model
class FunnelState(TypedDict):
    lead: dict
    messages: Annotated[list[BaseMessage], add_messages]
    score: int
    route: str
    email_body: str

# Step 1: Qualify lead
def qualify(state: FunnelState):
    prompt = f"Rate this lead {state['lead']} as JSON: fit, score(0-100), summary"
    res = llm.invoke([prompt]).content
    # For demo, fake parse
    score = 80 if state['lead']['years_exp'] > 2 else 50
    return {**state, "score": score}

# Step 2: Decide route
def decide(state: FunnelState):
    route = "book" if state["score"] >= 75 else "nurture"
    return {**state, "route": route}

# Step 3: Generate email
def gen_email(state: FunnelState):
    if state["route"] == "book":
        prompt = f"Write 3-sentence invite to {state['lead']['name']} to book call: {CALENDLY}"
    else:
        prompt = f"Write 3-line nurture email for {state['lead']['name']}."
    email = llm.invoke([prompt]).content
    return {**state, "email_body": email}

# Step 4: Send email
def send_email(state: FunnelState):
    msg = EmailMessage()
    msg["Subject"] = "Career Consultation" if state["route"] == "book" else "Resources for You"
    msg["From"], msg["To"] = SMTP_USER, state["lead"]["email"]
    msg.set_content(state["email_body"])
   
    return state

# DB checkpointer
conn = sqlite3.connect("funnel.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Build graph
graph = StateGraph(FunnelState)
graph.add_node("qualify", qualify)
graph.add_node("decide", decide)
graph.add_node("gen_email", gen_email)
graph.add_node("send_email", send_email)

graph.add_edge(START, "qualify")
graph.add_edge("qualify", "decide")
graph.add_edge("decide", "gen_email")
graph.add_edge("gen_email", "send_email")
graph.add_edge("send_email", END)

funnel = graph.compile(checkpointer=checkpointer)

# Demo run

lead = {"name": "Anjali", "email": "anjali@example.com", "years_exp": 3}
result = funnel.invoke({"lead": lead, "messages": []})
print(result["route"], "\n", result["email_body"])
