from datetime import timedelta
from openai import OpenAI
import pandas as pd
import streamlit as str
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai_like import OpenAILike
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode
import streamlit as st
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, MessageRole

# check if storage already exists



class Rag:
    def __init__(self):
        self.llm = Ollama(base_url="https://095kiew15yzv2e-8000.proxy.runpod.net",
            model="llama3.2-vision:90b")
    
    def load_transcript(self, timestamp: timedelta, lecture: str, lecture_id: str):
        transcript = pd.read_csv(f"data/{lecture}/{lecture_id}/transcripts/transcript.tsv", sep="\t")
        transcript['end'] = transcript['end'].map(lambda x: timedelta(milliseconds=x))
        transcript['start'] = transcript['start'].map(lambda x: timedelta(milliseconds=x))
        transcript = transcript[transcript['end'] <= timestamp]
        transcript['start_minutes'] = transcript['start'].map(lambda x: int(x.total_seconds() / 60))
        transcript = transcript.groupby('start_minutes').agg({'text': ' '.join}).reset_index()
        return transcript.to_dict(orient='records')
    

    def load_vectors(self, lecture: str, lecture_id: str):
        PERSIST_DIR = "./storage"
        embedding_model = get_embedding_model()
        if not os.path.exists(PERSIST_DIR):
            # load the documents and create the index
            
            documents = SimpleDirectoryReader(f"data/{lecture}/{lecture_id}/slides/text/").load_data()

            index = VectorStoreIndex.from_documents(
                documents=documents, embed_model=embedding_model
            )
            # store it for later
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            return index
        else:
                # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            return index
        
    def run(self, prompt: str, lecture:str, lecture_id: str):
        index = self.load_vectors(lecture=lecture, lecture_id=lecture_id)
        transcript = self.load_transcript(timedelta(minutes=120), lecture=lecture, lecture_id=lecture_id)

        index.insert_nodes([TextNode(text=segment['text'], metadata={'lecture': lecture_id, 'minute': segment['start_minutes'], 'type': 'transcript'}) for segment in transcript])
        query_engine = VectorIndexRetriever(index=index, similarity_top_k=5, embed_model=get_embedding_model())
        chunks = query_engine.retrieve(prompt)
        retrieved_text = "\n".join([chunk.get_content(metadata_mode="ALL") for chunk in chunks])
        
        return self.response_generator(user_input=prompt, retrieved_text = retrieved_text)


    def response_generator(self, user_input: str, retrieved_text: str):
        messages = [ChatMessage(role=MessageRole.USER, content=f"You are an assistant professor tasked with answering student questions based on the lecture transcript and slides given below. Always refer to the slide number and lecture minute. \n{retrieved_text}"),
        ChatMessage(role=MessageRole.USER, content=user_input)]
        return self.llm.chat(
            messages=messages,
            num_ctx=30000,
        ).message.content
    
@st.cache_resource
def get_embedding_model():
    Settings.embed_model = HuggingFaceEmbedding(model_name="baai/bge-small-en-v1.5")
    return HuggingFaceEmbedding(model_name="baai/bge-small-en-v1.5")