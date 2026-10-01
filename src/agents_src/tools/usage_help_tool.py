from crewai.tools import tool

@tool
def usage_help_tool(query: str) -> str:
    """
        Provides guidance to users about how to use AstraRAG.

        Use this tool when the user asks:
        - How to use AstraRAG
        - What can I ask?
        - How does this chatbot work?
        - What type of questions can I ask?
        - Help
        - Examples of questions
        - How to ask questions about the documents
    """

    return """
                AstraRAG is a document-based AI question-answering chatbot.

                How to use AstraRAG:

                1. Ask questions related to the documents available in the knowledge base.
                2. You can ask for explanations, summaries, comparisons, definitions, or specific information contained in the documents.
                3. You can ask follow-up questions based on the previous conversation.
                4. You can ask which documents/books are available before asking questions.
                5. If the requested information is not available in the knowledge base, AstraRAG will indicate that the information could not be found.

                Example questions:

                - What books are available?
                - What is the main topic of this document?
                - Explain the concept of biodiversity.
                - Summarize the important points from this chapter.
                - Compare the concepts discussed in two documents.
                - Explain this topic in simple language.
                - What does the document say about conservation?

                Tip:
                Start by asking "What books are available?" and then ask questions about the document you are interested in.

            """