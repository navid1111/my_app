from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://medium.com/@navidkamal/how-to-use-mongodb-with-typescript-and-what-might-break-957c1bf17755",
    header_template={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
)
docs = loader.load()
print(docs)
