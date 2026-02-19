import os
import streamlit as st
from typing import Annotated, TypedDict

# ── API kľúč – načítaj PRED inicializáciou LLM ────────────
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# ── LLM ───────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# ── Knowledge base (RAG) ──────────────────────────────────
loader = TextLoader("faq.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
chunks = splitter.split_documents(docs)

embeddings = FastEmbedEmbeddings()
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

# ── Stav agenta ───────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    sentiment: str
    category: str

# ── Uzol 1: Analyzuj správu ───────────────────────────────
def analyze(state: State):
    last_msg = state["messages"][-1].content
    response = llm.invoke([
        SystemMessage(content="""Analyze the customer message.
Reply ONLY in this exact format:
CATEGORY: [billing/technical/general]
SENTIMENT: [positive/neutral/negative]"""),
        HumanMessage(content=last_msg)
    ])
    text = response.content.lower()
    category = "billing" if "billing" in text else ("technical" if "technical" in text else "general")
    sentiment = "negative" if "negative" in text else ("positive" if "positive" in text else "neutral")
    return {"category": category, "sentiment": sentiment}

# ── Uzol 2: Odpoveď z knowledge base ─────────────────────
def answer(state: State):
    last_msg = state["messages"][-1].content
    context_docs = retriever.invoke(last_msg)
    context_text = "\n".join([d.page_content for d in context_docs])

    response = llm.invoke([
        SystemMessage(content=f"""Si priateľský zákaznícky support asistent.
Odpovedaj v slovenčine. Použi tieto informácie:

{context_text}

Ak odpoveď nevieš nájsť, povedz to úprimne."""),
        HumanMessage(content=last_msg)
    ])
    return {"messages": [response]}

# ── Uzol 3: Eskalácia na človeka ──────────────────────────
def escalate(state: State):
    msg = ("Rozumiem, že situácia nie je ideálna. "
           "Prepájam ťa na nášho kolegu, ktorý ti pomôže do 24 hodín. "
           "Kontaktuj nás aj priamo na support@firma.sk")
    return {"messages": [AIMessage(content=msg)]}

# ── Rozhodovacia logika ───────────────────────────────────
def route(state: State):
    if state["sentiment"] == "negative" or state["category"] == "billing":
        return "escalate"
    return "answer"

# ── Poskladaj graf ────────────────────────────────────────
builder = StateGraph(State)
builder.add_node("analyze", analyze)
builder.add_node("answer", answer)
builder.add_node("escalate", escalate)

builder.add_edge(START, "analyze")
builder.add_conditional_edges("analyze", route, {
    "answer": "answer",
    "escalate": "escalate"
})
builder.add_edge("answer", END)
builder.add_edge("escalate", END)

agent = builder.compile()
