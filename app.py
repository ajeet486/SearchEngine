import os
import streamlit as st
import wikipedia
from dotenv import load_dotenv
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
from langchain_classic.agents import initialize_agent, AgentType

load_dotenv()
wikipedia.set_user_agent("LangchainSearchApp/1.0 (your_email@example.com)")

## Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=500)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=500)
wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search = DuckDuckGoSearchRun(name="Search")

st.title("🔎 LangChain - Chat with search")
"""
In this example, we're using `StreamlitCallbackHandler` to display the thoughts and actions of an agent in an interactive Streamlit app.
Try more LangChain 🤝 Streamlit Agent examples at [github.com/langchain-ai/streamlit-agent](https://github.com/langchain-ai/streamlit-agent).
"""

## Sidebar
st.sidebar.title("Settings")
api_key = st.sidebar.text_input(
    "Enter your Groq API Key:", type="password",
    value=os.getenv("GROQ_API_KEY", ""),
)

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="What is machine learning?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-120b", streaming=True)
    tools = [search, arxiv, wiki]

    # search_agent = initialize_agent(
    #     tools,
    #     llm,
    #     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     handle_parsing_errors=True,
    # )
    search_agent = create_react_agent(llm, tools)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        result = search_agent.invoke(
          {"messages": st.session_state.messages},
                  {"callbacks": [st_cb]},)
                     
        response = result["messages"][-1].content
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)