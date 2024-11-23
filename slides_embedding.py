from pdf2image import convert_from_path
import os
from llama_index.embeddings.cohere import CohereEmbedding
import shutil
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.response.notebook_utils import display_source_node

         

def get_cohere_embeddings_and_store_to_nodes(folder_name):
    cohere_api_key = "WL54wHhH1dIJraDPkEru1dL5FEFKzCX847jvoPWD"
    os.environ["COHERE_API_KEY"] = cohere_api_key
    embed_model = CohereEmbedding(
    api_key=cohere_api_key,
    model_name="embed-english-v3.0",
    input_type="search_query",

)

    for item in os.listdir(folder_name):
        item_path = os.path.join(folder_name, item)
        if os.path.isfile(item_path) and item.lower().endswith(".txt"):
            with open (item_path,'r') as file:
                text = file.read()
                embeddings = embed_model.get_text_embedding(text)
                print(len(embeddings))
                print(embeddings[:5])

    documents = SimpleDirectoryReader("../../lecture2_txt").load_data()

    index = VectorStoreIndex.from_documents(
    documents=documents, embed_model=embed_model
)
    

    search_query_retriever = index.as_retriever()

    search_query_retrieved_nodes = search_query_retriever.retrieve(
        "What is a linear regression ?"
    )

    for n in search_query_retrieved_nodes:
        display_source_node(n, source_length=2000)
    
