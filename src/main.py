import streamlit as st
import os
from dotenv import load_dotenv
from typing import List
from typing_extensions import TypedDict

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from langchain_core.documents import Document
from langgraph.graph import START, END, StateGraph


# --- Page Configuration ---
st.set_page_config(
    page_title="RAG Agent with Web Search",
    page_icon="🤖",
    layout="wide"
)

# --- Load Environment Variables ---
load_dotenv()

# --- Constants ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
KNOWLEDGE_BASE_DIR = "knowledge-base"
PERSIST_DIRECTORY = "chroma_db"
LLM_MODEL_ID = "openai/gpt-oss-20b"


# --- RAG Functions ---
@st.cache_resource
def ingest_pdfs_into_vectordb():
    """Ingests PDFs into Chroma VectorDB."""
    documents = []
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        return 0
    
    for file_name in os.listdir(KNOWLEDGE_BASE_DIR):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(KNOWLEDGE_BASE_DIR, file_name)
            try:
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            except Exception as e:
                st.warning(f"Error loading {file_path}: {e}")
    
    if not documents:
        return 0
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, 
        chunk_overlap=CHUNK_OVERLAP
    )
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        texts, 
        embeddings, 
        persist_directory=PERSIST_DIRECTORY
    )
    vectorstore.persist()
    return len(documents)


@st.cache_resource
def create_retriever():
    """Creates a retriever from the Chroma DB."""
    if not os.path.exists(PERSIST_DIRECTORY):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY, 
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": 1})


# --- Graph State Definition ---
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    sender: str
    answer: str


# --- Graph Nodes ---
def router_node(state: GraphState) -> str:
    """The Router. Decides which tool to use next."""
    question = state["question"]
    routing_prompt = f"""You are an expert at routing user questions. 
    The **vectorstore** contains documents about **malaria symptoms, treatment, and prevention**. The **web_search** tool can access real-time information about latest Malaria information from WHO. 
    Based on the user's question, decide whether it's best to use the vectorstore or the web search.
    Question: "{question}" Respond with only 'vectorstore' or 'web_search'."""
    
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(routing_prompt)
    decision = response.content.strip().lower()
    return "web_search" if "web_search" in decision else "vectorstore"


def retrieve_node(state: GraphState) -> GraphState:
    """Retrieves documents from the vectorstore."""
    question = state["question"]
    retriever = create_retriever()
    if retriever is None:
        return {"documents": [], "sender": "retrieve_node"}
    retrieved_docs = retriever.invoke(question)
    return {"documents": retrieved_docs, "sender": "retrieve_node"}


def web_search_node(state: GraphState) -> GraphState:
    """Searches the web for information, then scrapes the content from the resulting URLs."""
    question = state["question"]
    
    tavily_search = TavilySearch(
        max_results=2, 
        search_depth="advanced", 
        include_domains=["who.int"]
    )
    search_results = tavily_search.invoke(question)
    
    scraped_docs = []
    if not search_results or "results" not in search_results:
        return {"documents": [], "sender": "web_search_node"}
    
    urls = [result.get("url") for result in search_results["results"] if result.get("url")]
    
    if not urls:
        return {"documents": [], "sender": "web_search_node"}
    
    for url in urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            scraped_docs.extend(docs)
        except Exception as e:
            st.warning(f"Error scraping {url}: {e}")
    
    return {"documents": scraped_docs, "sender": "web_search_node"}


def generate_node(state: GraphState) -> GraphState:
    """Generates an answer using the LLM."""
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join(doc.page_content for doc in documents)
    prompt = f"""You are an expert Q&A assistant. Use the following context to answer the user's question. If the context does not contain the answer, state that you cannot find the information. Be concise and helpful. Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"""
    llm = ChatGroq(temperature=0, model_name=LLM_MODEL_ID)
    response = llm.invoke(prompt)
    answer = response.content
    return {"answer": answer, "sender": "generate_node"}


# --- Build the Graph ---
@st.cache_resource
def build_graph():
    """Builds and compiles the workflow graph."""
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generate", generate_node)
    
    workflow.add_conditional_edges(
        START,
        router_node,
        {
            "vectorstore": "retrieve",
            "web_search": "web_search",
        },
    )
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()


# --- Streamlit UI ---
def main():
    st.title("🤖 RAG Agent with Web Search")
    st.markdown("Ask questions about malaria or any health-related topics!")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # PDF Ingestion Section
        st.subheader("📚 Knowledge Base")
        
        if os.path.exists(KNOWLEDGE_BASE_DIR):
            pdf_files = [f for f in os.listdir(KNOWLEDGE_BASE_DIR) if f.lower().endswith(".pdf")]
            st.info(f"Found {len(pdf_files)} PDF(s) in knowledge base")
            
            if pdf_files:
                with st.expander("View PDFs"):
                    for pdf in pdf_files:
                        st.text(f"📄 {pdf}")
        else:
            st.warning("Knowledge base directory not found")
        
        # Ingest Button
        if st.button("🔄 Ingest/Re-ingest PDFs", use_container_width=True):
            with st.spinner("Ingesting PDFs into vector database..."):
                doc_count = ingest_pdfs_into_vectordb()
                if doc_count > 0:
                    st.success(f"✅ Ingested {doc_count} documents!")
                    st.cache_resource.clear()
                else:
                    st.warning("No documents found to ingest")
        
        # Database Status
        st.subheader("💾 Database Status")
        if os.path.exists(PERSIST_DIRECTORY):
            st.success("Vector database exists")
        else:
            st.error("Vector database not found. Please ingest PDFs.")
        
        st.divider()
        
        # Information
        st.subheader("ℹ️ About")
        st.markdown("""
        This agent can:
        - 📖 Search local PDF documents
        - 🌐 Search the web for latest info
        - 🎯 Automatically route queries
        """)
    
    # Main Chat Interface
    st.divider()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source" in message:
                with st.expander("📚 Source"):
                    st.info(message["source"])
    
    # Chat input
    if question := st.chat_input("Ask a question about malaria or health topics..."):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Build and run the graph
                    app = build_graph()
                    inputs = {"question": question}
                    
                    # Stream the output
                    result = None
                    for output in app.stream(inputs, stream_mode="values"):
                        result = output
                    
                    if result and "answer" in result:
                        answer = result["answer"]
                        source_info = result.get("sender", "unknown")
                        
                        st.markdown(answer)
                        
                        # Show source information
                        if "documents" in result and result["documents"]:
                            sources = []
                            for doc in result["documents"]:
                                if "source" in doc.metadata:
                                    sources.append(doc.metadata["source"])
                            
                            if sources:
                                source_text = "Sources:\n" + "\n".join(f"- {s}" for s in sources)
                                with st.expander("📚 View Sources"):
                                    st.text(source_text)
                                
                                # Add to chat history with source
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer,
                                    "source": source_text
                                })
                            else:
                                st.session_state.messages.append({
                                    "role": "assistant", 
                                    "content": answer
                                })
                        else:
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer
                            })
                    else:
                        error_msg = "I couldn't generate an answer. Please try again."
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
                
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()