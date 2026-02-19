import streamlit as st
from langchain_core.messages import HumanMessage
from agent import agent  # importuje agent z agent.py

st.set_page_config(page_title="Customer Support", page_icon="🤖")
st.title("🤖 Zákaznícky support")

if "history" not in st.session_state:
    st.session_state.history = []

# Zobraz históriu
for role, msg in st.session_state.history:
    st.chat_message("user" if role == "ty" else "assistant").write(msg)

# Input
user_input = st.chat_input("Napíš svoju otázku...")
if user_input:
    with st.spinner("Spracovávam..."):
        result = agent.invoke({
            "messages": [HumanMessage(content=user_input)],
            "sentiment": "",
            "category": ""
        })
    response = result["messages"][-1].content

    st.session_state.history.append(("ty", user_input))
    st.session_state.history.append(("agent", response))

    st.chat_message("user").write(user_input)
    st.chat_message("assistant").write(response)
