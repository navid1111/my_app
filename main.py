from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()
loader = WebBaseLoader(
    "https://medium.com/@navidkamal/how-to-use-mongodb-with-typescript-and-what-might-break-957c1bf17755",
    header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
)
# First we read our data from website
# Then we load it into our document loader

docs = loader.load()
# print(docs)

#  we can't give this directly to our llm models because every model has a limit on how many tokens it can process.
# So we need to split it into smaller chunks.

text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

documents=text_splitter.split_documents(docs)

# print(documents)

emeddings = OpenAIEmbeddings()

# now we have to convert these documents into vectors
vectorestore_db= FAISS.from_documents(documents,emeddings)
print(vectorestore_db)