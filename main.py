from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o")

# Load documents from web
loader = WebBaseLoader(
    "https://medium.com/@navidkamal/how-to-use-mongodb-with-typescript-and-what-might-break-957c1bf17755",
    header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
)

# Load the documents
docs = loader.load()
print(f"Loaded {len(docs)} documents")

# Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
documents = text_splitter.split_documents(docs)
print(f"Split into {len(documents)} chunks")

# Create embeddings
embeddings = OpenAIEmbeddings()

# Create vector store
vectorstore_db = FAISS.from_documents(documents, embeddings)
print("Vector store created successfully")

# Define the query
query = "what are the issues we will face while using mongodb with typescript?"

# Test similarity search
result = vectorstore_db.similarity_search(query)
print(f"Found {len(result)} similar documents")

# Method 1: Using create_retrieval_chain (fixed version)
print("\n" + "="*50)
print("METHOD 1: Using create_retrieval_chain")
print("="*50)

# Create prompt template with correct variable names
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context:

Context: {context}

Question: {input}

Answer:""")

# Create document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create retriever
retriever = vectorstore_db.as_retriever()

# Create retrieval chain
retriever_chain = create_retrieval_chain(retriever, document_chain)

try:
    result = retriever_chain.invoke({"input": query})
    print("ANSWER:")
    print(result["answer"])
    
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(result["context"]):
        print(f"Document {i+1}:")
        print(doc.page_content[:200] + "...")
        print("-" * 30)
        
except Exception as e:
    print(f"Method 1 failed: {e}")
    
    # Method 2: Manual RAG chain (alternative approach)
    print("\n" + "="*50)
    print("METHOD 2: Manual RAG Chain")
    print("="*50)
    
    # Create a custom prompt template
    template = """Answer the following question based only on the provided context:

    Context: {context}

    Question: {question}

    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create a manual RAG chain
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    # Create the chain manually
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    # Invoke the chain
    result = rag_chain.invoke(query)
    print("ANSWER:")
    print(result.content)
    
    # Show retrieved documents
    retrieved_docs = retriever.invoke(query)
    print("\nSOURCE DOCUMENTS:")
    for i, doc in enumerate(retrieved_docs):
        print(f"Document {i+1}:")
        print(doc.page_content[:200] + "...")
        print("-" * 30)