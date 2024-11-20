# Infobridge
InfoBridge is a Streamlit-based tool for extracting summaries and answering queries from news article URLs using LangChain and Ollama's LLaMA 3.2 model. The project leverages embeddings, vector stores, and retrieval-based QA to provide accurate results.

## Features
<br>**Data Processing:**
- load the url and using Unstructuredurlloader  and split them into chunks
  
<br>**Embedding and vector stores**
- Generate the embedding using ollama embeddings and store them in FAISS vector store.
- Save the vector store locally for faster querrying
  
<br>**Querrying:**
-  The RetrievalQA chain retrieves relevant content and generates answers to user queries.



