from crewai import Task
from pydantic import BaseModel

from src.agents_src.agents.question_answer_agent import qa_agent


# Define the expected structure of the answer for the question answering task, structured output
class AnswerStructure(BaseModel):
    answer: str
    sources: list[str]
    tool_used: str
    rationale: str


qa_task = Task(
    agent=qa_agent,
    name = "Question Answering Task",
    description="""
     Answer the user query "{user_query}" using a Retrieval-Augmented Generation (RAG) pipeline.
    chat_history: "{chat_history}"
    
    Instructions:
      - First understand the user's intent.

      - If the user is asking how to use AstraRAG, what they can ask,
        how the chatbot works, or requests examples/help,
        use the usage_help_tool.

      - If the user is asking which books, PDFs, or documents are available,
        use the list_documents_tool.

      - If the user is asking a question about the content of the documents,
        use the rag_query_tool.

      - Do not use the RAG query tool to list all available documents.

      - Prioritize evidence that directly addresses the query.

      - Synthesize a clear and accurate answer grounded in the retrieved
        sources or chat history.

      - If the query cannot be answered from the knowledge source or
        chat history, do not generate your own response.

      - Instead, clearly state that the knowledge source does not contain
        the required information.

      - Provide transparency by including the tool used and relevant
        sources.
    """,
     expected_output="""
    A structured JSON object with the following fields:
    {
      "answer": "Direct response to the query (1-3 paragraphs, clear and accurate). 
                 If no answer is found, return: 'The knowledge source does not contain the required information.'",
      "sources": ["List of document titles, sections, or citations used (empty list if none)"],
      "tool_used": "Name of the retrieval/analysis tool invoked (e.g., RAG Retriever, VectorDB, ChatHistory, etc.)",
      "rationale": "Brief explanation of why this answer was chosen, or why no relevant information was found"
    }
    """,
    output_pydantic=AnswerStructure,
)
