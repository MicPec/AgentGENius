from numpy import tile
import streamlit as st
import nest_asyncio
from dotenv import load_dotenv
from agentgenius.core import AgentGENius
from agentgenius.tools import ToolSet
from pydantic_ai import RunContext
import json
import pandas as pd

# Initialize environment
nest_asyncio.apply()
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "agent" not in st.session_state:
    # Create basic tools
    def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Get the current datetime as a string in the specified python format."""
        from datetime import datetime

        return datetime.now().strftime(format)

    tools = ToolSet(get_current_datetime)
    st.session_state.agent = AgentGENius(name="chat_agent", model="openai:gpt-4o-mini", toolset=tools)
    st.session_state.agent.agent_store.load_agents()


def get_agent_stats():
    """Get current agent statistics"""
    agent = st.session_state.agent
    stats = {
        "Model": [agent.model],
        "Tools": [", ".join(tool.__name__ for tool in agent.toolset)],
        "Agents": [", ".join(agent.agent_store.agents.keys())],
    }
    return pd.DataFrame(stats)


def display_sidebar():
    """Display agent statistics in sidebar"""
    st.sidebar.title("🤖 Statistics")
    stats_df = get_agent_stats()

    st.sidebar.subheader("Agent Statistics")
    # Display stats in a transposed table for better readability
    st.sidebar.dataframe(
        stats_df.T,
        column_config={"0": st.column_config.ListColumn("Value", width="big")},
        hide_index=False,
        use_container_width=True,
    )

    # Display available tools in a separate table
    tools_data = [
        {
            "Tool": tool.__name__,
            "Description": tool.__doc__.replace("\n", " ").replace("    ", "") or "No description available",
        }
        for tool in st.session_state.agent.toolset
    ]
    st.sidebar.subheader("Available Tools")
    st.sidebar.dataframe(
        pd.DataFrame(tools_data),
        column_config={
            "Tool": st.column_config.TextColumn("Tool", width="small"),
            "Description": st.column_config.TextColumn("Description", width="medium"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # Display available agents in a separate table
    agents_data = []
    for name, agent in st.session_state.agent.agent_store.agents.items():
        agents_data.append(
            {
                "Name": name,
                "Model": agent.model,
                "Prompt": agent.get_system_prompt(),
                "Tools ": [tool for tool in agent.toolset.list],
            }
        )

    if agents_data:
        st.sidebar.subheader("Available Agents")
        st.sidebar.dataframe(
            pd.DataFrame(agents_data),
            column_config={
                "Name": st.column_config.TextColumn("Name", width="small", pinned=True),
                "Model": st.column_config.TextColumn("Model", width="small"),
                "Tools": st.column_config.ListColumn("Tools", width="small"),
            },
            hide_index=True,
            use_container_width=True,
        )


def response_details(messages):
    tabs = st.tabs([f"{i+1}: {msg['kind']}" for i, msg in enumerate(messages)])

    for tab, msg in zip(tabs, messages):
        with tab:
            st.write(msg)


def display_chat():
    """Display chat interface"""
    st.title("AgentGENius Chat")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.agent.run_sync(prompt, message_history=st.session_state.message_history)
                st.session_state.message_history += response.new_messages()
                if len(st.session_state.message_history) > 20:
                    st.session_state.message_history = st.session_state.message_history[-20:]

                st.session_state.messages.append({"role": "assistant", "content": response.data})
                st.markdown(response.data)
                with st.expander("See detailed response"):
                    response_details(json.loads(response.new_messages_json()))


def main():
    st.set_page_config(
        page_title="AgentGENius Chat",
        page_icon="🤖",
        layout="wide",
    )

    display_sidebar()
    display_chat()


if __name__ == "__main__":
    main()
