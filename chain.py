import box
import yaml
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceBgeEmbeddings

# Load the configuration file
with open('config.yml', 'r', encoding='utf8') as f:
    config = box.Box(yaml.safe_load(f))

# Load the embeddings using HuggingFace
embeddings = HuggingFaceBgeEmbeddings(
    model_name=config.data.embeddings,
    model_kwargs={'device': 'cpu'}
)

# Load the persisted FAISS vectorstore from disk
vectorstore = None
vectorstore = FAISS.load_local(
    folder_path=config.data.faiss.path,
    embeddings=embeddings
)

# Get the retriever with maximum 2 results
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

# Create the LLM chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
print(qa_chain)

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response['source_documents']:
        print(source.metadata['source'])
        
# Full example
query = "Who can purchase Manulife ReadyBuilder?"
llm_response = qa_chain(query)
process_llm_response(llm_response)

