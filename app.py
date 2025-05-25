import streamlit as st
from typing import Iterator
from agno.run.response import RunResponse
from agents import get_agent_team

# Set page config
st.set_page_config(
    page_title="Canadian Agents",
    page_icon="🇨🇦",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

def parse_stream(stream):
    for chunk in stream:
        if chunk.content is not None:
            yield chunk.content

# App title and description
st.title("🇨🇦 Canadian Agents")
st.caption("Get help finding, supporting, and learning about Canadian businesses.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🍁" if message["role"] == "assistant" else "💁‍♀️"):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("How can I help you support Canadian businesses?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user", avatar="💁‍♀️"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant", avatar="🍁"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Thinking..."):
            agent_team = get_agent_team()
            
            stream: Iterator[RunResponse] = agent_team.run(
                prompt, 
                stream=True, 
                auto_invoke_tools=True,
                user_id="ava",
            )
            response = st.write_stream(parse_stream(stream))

        st.session_state.messages.append({"role": "assistant", "content": full_response}) 