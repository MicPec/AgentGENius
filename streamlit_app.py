import json

import nest_asyncio
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import RunContext

from agentgenius import AgentDef, Task, TaskDef
from agentgenius.builtin_tools import get_datetime, get_location_by_ip, get_user_ip
from agentgenius.tools import ToolSet

# Initialize environment
nest_asyncio.apply()
load_dotenv()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "message_history" not in st.session_state:
    st.session_state.message_history = []
if "agent" in st.session_state:
    del st.session_state.agent

if "planner" not in st.session_state:
    # Define a planner task
    planner_task = Task(
        task=TaskDef(name="planner", question="make a short plan how to archive this task", priority=1),
        agent_def=AgentDef(
            model="openai:gpt-4o",
            name="planner",
            system_prompt="""You are a planner. your goal is to make a step by step plan for other agents. 
            Do not answer the user questions. Just make a very short plan how to do this. 
            AlWAYS MAKE SURE TO ADD APPROPRIATE TOOLS TO THE PLAN. You can get the list of available tools by calling 'get_available_tools'.
            Efficiently is a priority, so don't waste time on things that are not necessary.
            LESS STEPS IS BETTER (up to 3 steps), so make it as short as possible.
            Tell an agent to use the tools if available. Use the users language""",
            params={
                "result_type": Task,
                "retries": 5,
            },
        ),
        toolset=ToolSet([get_datetime, get_user_ip, get_location_by_ip]),
    )
    st.session_state.planner = planner_task


def get_agent_stats():
    """Get current planner task statistics"""
    planner = st.session_state.planner
    stats = {
        "Task Name": [planner.task.name],
        "Question": [planner.task.question],
        "Priority": [planner.task.priority],
        "Tools": [", ".join(tool.__name__ for tool in planner.toolset)],
    }
    return pd.DataFrame(stats)


def display_sidebar():
    """Display planner task statistics in sidebar"""
    st.sidebar.title("🤖 Statistics")
    stats_df = get_agent_stats()

    st.sidebar.subheader("Planner Task Statistics")
    # Display stats in a transposed table for better readability
    st.sidebar.dataframe(
        stats_df.T,
        column_config={"0": st.column_config.TextColumn("Value", width="big")},
        hide_index=False,
        use_container_width=True,
    )

    # Display available tools in a separate table
    tools_data = [
        {
            "Tool": tool.__name__,
            "Description": tool.function.__doc__.replace("\n", " ").replace("    ", "") or "No description available",
        }
        for tool in st.session_state.planner.toolset
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


def response_details(messages, usage):
    tabs = st.tabs([f"{i+1}: {msg['kind']}" for i, msg in enumerate(messages)] + ["Usage Summary"])
    messages.append(
        {
            "requests": usage.requests,
            "request_tokens": usage.request_tokens,
            "response_tokens": usage.response_tokens,
            "total_tokens": usage.total_tokens,
            "details": usage.details,
        }
    )
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
                response = st.session_state.planner.run_sync(prompt, message_history=st.session_state.message_history)
                st.session_state.message_history += response.new_messages()
                if len(st.session_state.message_history) > 20:
                    st.session_state.message_history = st.session_state.message_history[-20:]

                st.session_state.messages.append({"role": "assistant", "content": response.data})
                st.markdown(response.data)
                with st.expander("See detailed response"):
                    response_details(json.loads(response.new_messages_json()), response.usage())


def display_planner():
    """Display planner interface"""
    st.title("AgentGENius Planner")
    st.write(st.session_state.messages)
    if st.button("Run Planner"):
        if st.session_state.messages:
            last_user_message = st.session_state.messages[-2]["content"]
        else:
            last_user_message = "Hello"
        result = st.session_state.planner.run_sync(last_user_message)
        result_data = result.data
        st.write(result_data)
        st.write(result_data.run_sync().data)


def main():
    st.set_page_config(
        page_title="AgentGENius Chat",
        page_icon="🤖",
        layout="wide",
    )

    display_sidebar()
    display_chat()
    display_planner()


if __name__ == "__main__":
    main()
