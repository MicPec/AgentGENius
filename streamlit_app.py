#!/usr/bin/env python3
"""Streamlit chat application using AgentGENius."""

import asyncio
import json
from pathlib import Path

import logfire
import streamlit as st
from dotenv import load_dotenv
from rich import print

from agentgenius.main import AgentGENius

# Load environment variables and configure logging
load_dotenv()
logfire.configure(send_to_logfire="if-token-present")

# Page config
st.set_page_config(
    page_title="AgentGENius Chat",
    page_icon=":robot_face:",
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


def display_task_statistics(history_file: str = "task_history.json"):
    """Display task statistics from history file"""
    history_path = Path("history") / history_file
    if not history_path.exists():
        st.error(f"History file {history_file} not found")
        return

    history_data = json.loads(history_path.read_text(encoding="utf-8"))
    st.markdown(f"**History Length:** {len(history_data['items'])} / {history_data['max_items']} items")

    # History selection
    history_items = [f"{i + 1}. {item['user_query']}" for i, item in enumerate(history_data["items"])][::-1]
    selected_index = st.selectbox("Select query from history:", history_items)
    if selected_index:
        current_item = history_data["items"][int(selected_index.split(".")[0]) - 1]

        st.markdown("---")
        st.markdown(f"**User Query:** {current_item['user_query']}")
        if current_item["tasks"]:
            st.markdown(f"**Number of subtasks:** {len(current_item['tasks'])}")
            for task in current_item["tasks"]:
                with st.expander(f"Task: {task['query']}"):
                    st.markdown(f"**Result:** {task['result']}\n\n---")
                    if task["tool_results"]:
                        st.markdown("**Tools used:**")

                        tabs = st.tabs([f"🔧 {tool['tool']}" for tool in task["tool_results"]])
                        for tab, tool in zip(tabs, task["tool_results"]):
                            with tab:
                                st.markdown("**Arguments:**")
                                st.code(tool["args"], language="json")
                                st.markdown("**Result:**")
                                st.code(tool["result"])
        else:
            st.info("Direct response (no subtasks)")


def main():
    """Main application function."""
    st.title("💬 AgentGENius Chat")

    # Initialize session state for statistics container
    if "stats_container" not in st.session_state:
        st.session_state.stats_container = None

    # Sidebar
    with st.sidebar:
        st.title("Settings")
        if st.button("Clear Chat", key="clear"):
            clear_chat()

        st.markdown("---")
        # Create a container for statistics that we can update
        stats_container = st.container()
        with stats_container:
            if st.session_state.stats_container:
                display_task_statistics()

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
                    # Update statistics after response
                    st.session_state.stats_container = True
                    # Force refresh of the sidebar
                    st.rerun()
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})


if __name__ == "__main__":
    main()
