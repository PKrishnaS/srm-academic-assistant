from chunking import process_directory
from langchain.docstore.document import Document

def load_docs(directory):
    results = process_directory(directory)
    documents = []
    for filename, chunks in results.items():
        for i, chunk in enumerate(chunks):
            metadata = {
                "source": filename,
                "page": i,  # Using chunk index as page number
                "headers": chunk['headers']
            }
            documents.append(Document(page_content=chunk['content'], metadata=metadata))
    return documents
