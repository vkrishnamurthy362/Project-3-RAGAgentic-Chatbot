import chromadb
from crewai.tools import tool

from src.agents_src.config.agent_settings import AgentSettings

@tool
def list_documents_tool(query: str="") -> str:
    """
        Lists all unique documents available in the knowledge base.

        Use this tool when the user asks:
        - What documents are available?
        - List all documents
        - Show me the documents
        - List all PDFs
        - what can I ask questions about?
    """

    settings = AgentSettings()

    db = chromadb.PersistentClient(path=settings.VECTOR_STORE_DIR)

    collection = db.get_or_create_collection(settings.COLLECTION_NAME)

    result = collection.get(include=["metadatas"])

    metadatas = result.get("metadatas",[])

    document_names = set()

    for metadata in metadatas:
        if metadata:
            file_name = metadata.get("file_name")

            if file_name:
                document_names.add(file_name)

    documents = sorted(document_names)

    if not documents:
        return "No documents are currently available in the knowledge base."

    response = (
        f"The following {len(documents)} document(s) are available "
        "for you to ask questions about:\n\n"
    )

    for index, document in enumerate(documents, start=1):
        response += f"{index}. {document}\n"

    response += (
        "\nYou can now ask me questions about any of these documents."
    )

    return response