import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from typing import Iterator
from agno.storage.sqlite import SqliteStorage
from agno.knowledge.csv import CSVKnowledgeBase
from agno.vectordb.chroma import ChromaDb
from agno.run.response import RunResponse
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools

DEBUG_MODE = True

# Set page config
st.set_page_config(
    page_title="Canadian Agents",
    page_icon="🇨🇦",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

canadian_shopify_businesses_knowledge_base = CSVKnowledgeBase(
    path="data/shop_canada_data.csv", # from here: https://github.com/parker84/shop-canada
    vector_db=ChromaDb(collection="canadian_shopify_businesses"),
    num_documents=10,  # Number of documents to return on search
)

@st.cache_resource
def get_agent_team():
    business_finder_agent = Agent(
        name="Canadian Business Finder",
        role="Find and recommend Canadian businesses",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Focus on finding Canadian businesses and products",
            "Include business location and contact information when available",
            "Highlight unique Canadian aspects of the businesses",
            "Consider different regions of Canada",
            "Include both online and physical businesses",
            "Always include sources (and link out to them)",
            "Only include businesses that are Canadian"
        ],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    # TODO: browser control tool

    news_agent = Agent(
        name="Canadian Business News",
        role="Provide updates on Canadian business news and trends",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Focus on Canadian business news and developments",
            "Include information about government support programs",
            "Highlight success stories of Canadian businesses",
            "Track industry trends in Canada",
            "Include relevant statistics and data",
            "Provide context about the Canadian business landscape",
            "Always include sources (and link out to them)"
        ],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    shopify_finder_agent = Agent(
        name="Shopify Business Finder",
        role="Find and recommend Canadian Shopify businesses",
        model=OpenAIChat(id="gpt-4o-mini"),
        knowledge=canadian_shopify_businesses_knowledge_base,
        search_knowledge=True,
        # tools=[DuckDuckGoTools()], # comment out to use explicitly use knowledge base
        instructions=[
            "Focus on finding Canadian Shopify businesses from the knowledge base",
            "Only use the knowledge base to find relevant businesses and data",
            "Do not include any other businesses in your output that are not in the knowledge base",
            "Always include links out to the shopify stores and the shop app page",
            "Always include a description of each business and the unique Canadian aspect of the businesses",
            "Always rank businesses on your output lists by volume of ratings in the knowledge base",
            "Always include volume of ratings and average rating in your output table",
            "Do not include any other businesses in your output that are not in the knowledge base"
        ],
        debug_mode=DEBUG_MODE,
    )
    shopify_finder_agent.knowledge.load(recreate=False)

    support_agent = Agent(
        name="Canadian Business Support",
        role="Provide information about supporting Canadian businesses",
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Explain different ways to support Canadian businesses",
            "Provide information about Canadian business directories",
            "Highlight the benefits of buying Canadian",
            "Suggest ways to promote Canadian businesses",
            "Include information about Canadian business associations",
            "Provide resources for business owners",
            "Explain government support programs",
        ],
        show_tool_calls=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    analysis_agent = Agent(
        name="Canadian Business Analysis",
        role="Analyze Canadian business trends and opportunities",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, company_info=True)],
        instructions=[
            "Analyze Canadian business sectors and industries",
            "Provide insights into Canadian market trends",
            "Compare Canadian businesses with international competitors",
            "Highlight growth opportunities for Canadian businesses",
            "Include relevant economic data and statistics",
            "Use tables and charts to present data clearly",
            "Always include sources (and link out to them)"
        ],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

    agent_storage: str = "tmp/agents.db"

    agent_team = Agent(
        team=[business_finder_agent, news_agent, support_agent, analysis_agent, shopify_finder_agent],
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions=[
            "Focus on supporting and promoting Canadian businesses",
            "Provide accurate and up-to-date information",
            "Include specific examples and recommendations",
            "Consider different regions and industries in Canada",
            "Highlight the benefits of supporting Canadian businesses",
            "Use tables and charts to present data clearly",
            "Always include sources (and link out to them)"
        ],
        show_tool_calls=True,
        debug_mode=DEBUG_MODE,
        storage=SqliteStorage(table_name="canadian_business_agent", db_file=agent_storage),
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
    )
    return agent_team

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
            )
            response = st.write_stream(parse_stream(stream))

        st.session_state.messages.append({"role": "assistant", "content": full_response}) 