#!/usr/bin/env python3
"""Streamlit chat application using AgentGENius."""

import asyncio
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from rich import print
import logfire

from agentgenius.main import AgentGENius

# Load environment variables and configure logging
load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

# Page config
st.set_page_config(
    page_title="AgentGENius Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "agent" not in st.session_state:
        try:
            st.session_state.agent = AgentGENius(model="openai:gpt-4o")
        except Exception as e:
            st.error(f"Failed to initialize agent: {str(e)}")
            st.session_state.agent = None


async def get_agent_response(prompt: str) -> str:
    """Get response from the agent asynchronously.

    Args:
        prompt: User's input message

    Returns:
        Agent's response as string
    """
    try:
        return await st.session_state.agent.ask(prompt)
    except Exception as e:
        print(f"Error getting response: {e}")
        return f"I apologize, but I encountered an error: {str(e)}"


def clear_chat():
    """Clear chat history."""
    st.session_state.messages = []


def main():
    """Main application function."""
    st.title("💬 AgentGENius Chat")

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear Chat", key="clear"):
            clear_chat()

        st.markdown("---")
        st.markdown("""
        ### About
        AgentGENius is an advanced AI assistant that can help you with:
        - Answering questions
        - Writing and analyzing code
        - Solving problems
        - And much more!
        """)

    # Initialize session state
    initialize_session_state()

    # Check if agent is initialized
    if not st.session_state.agent:
        st.error("Failed to initialize the chat agent. Please check your configuration and try again.")
        return

    # Main chat interface
    st.markdown("""
    Welcome to AgentGENius Chat! Enter your message below to start the conversation.
    """)

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to discuss?", key="chat_input"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Run async function in sync context
                    response = asyncio.run(get_agent_response(prompt))
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
