from crewai import Agent

from src.agents_src.tools.rag_qa_tool import rag_query_tool
from src.agents_src.tools.list_documents_tool import list_documents_tool
from src.agents_src.tools.usage_help_tool import usage_help_tool
from src.agents_src.llm.get_llm import get_llm_for_agent


name = "Question Answer Agent"
llm = get_llm_for_agent(name)


qa_agent = Agent(
    role="Question Answer Agent",
    llm=llm,
    tools=[rag_query_tool,
           usage_help_tool,
           list_documents_tool],
    goal="""
            Understand the user's intent and provide accurate, helpful responses.

            Use the appropriate available tool:

            - usage_help_tool for questions about how to use AstraRAG
            - list_documents_tool for identifying available documents
            - rag_query_tool for answering questions from document content

            Keep responses clear, factual, user-friendly, and grounded in the
            available knowledge base.
        """,
    backstory=" You are a knowledgeable analyst who has spent years helping people find clarity in large"
              " document collections. You specialize in surfacing the most relevant evidence and turning it into clear,"
              " reliable answers. You value precision and transparency, always grounding responses in sources so"
              " users can trust the insights you provide.",
    verbose=True
)
