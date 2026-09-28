FROM python:3.11-slim

WORKDIR /app

# INSTALL SYSTEM DEPENDENCIES
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# COPY REQUIREMENTS AND INSTALL
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY APPLICATION CODE
COPY src/ src/
COPY docs_dir/ docs_dir/
COPY start.sh ./start.sh

# MAKE START SCRIPT EXECUTABLE
RUN chmod +x ./start.sh

# EXPOSE BACKEND AND FRONTEND PORTS
EXPOSE 8000
EXPOSE 8501

# SET ENVIRONMENT VARIABLES (CAN BE OVERRIDDEN AT RUNTIME)
ENV GROQ_API_KEY="YOUR_GROQ_API_KEY"
ENV DOCUMENTS_DIR="/app/docs_dir"
ENV VECTOR_STORE_DIR="/app/doc_vector_store"
ENV COLLECTION_NAME="document_collection"
ENV MODEL_NAME="openai/gpt-oss-20b"
ENV MODEL_TEMPERATURE=0.0
ENV CHAT_ENDPOINT_URL="http://localhost:8000/chat/answer"

# RUN ALL SERVICES USING START.SH
CMD ["/app/start.sh"]
