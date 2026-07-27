from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )

        self.vector_store = None

    def add_text_documents(self, text: str, source: str = ""):
        from langchain_core.documents import Document
        doc = Document(page_content=text, metadata={"source": source})
        return self.add_documents([doc])

    def add_documents(self, docs):
        chunks = self.text_splitter.split_documents(docs)

        if not self.vector_store:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            new_vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.merge_from(new_vector_store)

    def retrieve(self, query: str, k: int = 10, filter_type: str = ""):
        if not self.vector_store:
            return []
        docs = self.vector_store.similarity_search(query, k=k)
        formatted_chunks = []
        for doc in docs:
            src = doc.metadata.get("source", "")
            header = f"[Source: {src}]\n" if src else ""
            formatted_chunks.append(f"{header}{doc.page_content}")
        return "\n\n".join(formatted_chunks)


rag_pipeline = RAGPipeline()