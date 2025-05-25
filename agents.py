import streamlit as st
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.storage.sqlite import SqliteStorage
from agno.knowledge.csv import CSVKnowledgeBase
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.chroma import ChromaDb
from agno.team.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory
import os

DEBUG_MODE = os.getenv("DEBUG_MODE", "True").lower() == "true"

canadian_shopify_businesses_knowledge_base = CSVKnowledgeBase(
    path="data/shop_canada_data.csv", # from here: https://github.com/parker84/shop-canada
    vector_db=ChromaDb(collection="canadian_shopify_businesses"),
    num_documents=10,  # Number of documents to return on search
    # chunking_strategy=AgenticChunking(), # agentic chunking: https://docs.agno.com/chunking/agentic-chunking
)

@st.cache_resource
def get_agent_team():
    search_agent = Agent(
        name="Search Agent",
        role="Search the web for information",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[DuckDuckGoTools()],
        instructions=[
            "Search the web for information",
            "Include sources (and link out to them)",
            "Always include sources (and link out to them)",
            "But don't just include the sources, pull out the relevant information from the sources"
        ],
        show_tool_calls=True,
    )

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
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )
    shopify_finder_agent.knowledge.load(recreate=False)

    reasoning_agent = Agent(
        name="Reasoning Agent",
        role="Reasoning Agent",
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[
            ReasoningTools(
                think=True,
                analyze=True,
                add_instructions=True,
                add_few_shot=True,
            ),
        ],
        show_tool_calls=True,
        add_datetime_to_instructions=True,
        markdown=True,
        debug_mode=DEBUG_MODE,
    )

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
        add_datetime_to_instructions=True,
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

    # Database file for memory and storage
    db_file = "tmp/agent.db"

    # Initialize memory.v2
    memory = Memory(
        # Use any model for creating memories
        model=OpenAIChat(id="gpt-4o-mini"),
        db=SqliteMemoryDb(table_name="user_memories", db_file=db_file),
    )

    agent_team = Team(
        name="Canadian Business Agent Team",
        description="A team of agents that can help you find and support Canadian businesses",
        members=[
            # search_agent, business_finder_agent, news_agent, support_agent, analysis_agent, 
            # shopify_finder_agent, reasoning_agent
            search_agent
        ],
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
        add_datetime_to_instructions=True,
        markdown=True,
        show_members_responses=True,
        # ----------memory----------
        # adding previous 5 questions and answers to the prompt
        # read more here: https://docs.agno.com/memory/introduction
        storage=SqliteStorage(table_name="agent_sessions", db_file=db_file),
        enable_team_history=True,
        num_history_runs=5,
        # adding agentic memory: "With Agentic Memory, The Agent itself creates, updates and deletes memories from user conversations."
        # read more here: https://docs.agno.com/memory/memory
        enable_agentic_memory=True,
        memory=memory,
    )
    return agent_team


# I'm a vegetarian
# top restaurants in toronto

