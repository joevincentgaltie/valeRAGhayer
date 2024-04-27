__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain.document_loaders import CSVLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_chroma import Chroma
import os


from rag.utils import mapper_partis, assistant_mapper_party

#from st_files_connection import FilesConnection

# Create connection object and retrieve file contents.
# Specify input format is a csv and to cache the result for 600 seconds.
# conn = st.connection('s3', type=FilesConnection)
# df = conn.read("testbucket-jrieke/myfile.csv", input_format="csv", ttl=600)



#load_dotenv()
api_key = st.secrets["API_MISTRAL"]
hugging_face_token = st.secrets["HUGGING_FACE"]
from huggingface_hub import login
login(token=hugging_face_token)

#os.getenv("API_MISTRAL")


streamlit_style = """
			<style>
			html, body, [class*="css"]  {
			font-family: 'Impact', sans-serif;
			}
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)


st.markdown("""
<style>
.big-font {
    font-size:50px !important;
    font-family:Impact;
}
            
.medium-font {
    font-size:30px ;
    font-family:Impact;
}
</style>
""", unsafe_allow_html=True)
st.markdown('<p class="big-font"> DemocracIA </p>', unsafe_allow_html=True)
st.markdown('<p class="medium-font"> Nous votons, eux aussi 👀  </p>', unsafe_allow_html=True)




# Afficher le titre avec la police personnalisée
#st.title('ValeRAG Hayer')

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
model = ChatMistralAI(mistral_api_key=api_key,
                      temperature=.2, 
                      stream = True, 
                      safe_prompt = True)


@st.cache_resource 
def load_data() : 
    with st.spinner(text="Chargement de la base de données, patience !"):
        loader = CSVLoader('db/all_french_explanations.csv',metadata_columns=["party", "number","name", "source", "source_date","orientation"], encoding="utf-8")
        documents =loader.load()
        db = Chroma.from_documents(documents[:10000], embeddings, persist_directory="../chroma_db")
        return db

db = load_data()





#db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)




st.warning('Version demo : nous ne saurions être tenus responsables si votre député(e) préféré(e) est diffamé(e)', icon="⚠️")

party = mapper_partis[st.selectbox(label = "Parti politique" , options=mapper_partis.keys())]

retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, 'filter' :  {"party": party}}
)

# prompt = ChatPromptTemplate.from_template("""Resume only in french language the political position of the MPs of the party {party} on the following subject based only on the provided context:

# <context>
# {context}
# </context>

# Question: {input}""")

prompt = ChatPromptTemplate.from_template("""Résume, en français uniquement, la position des députés européens du parti {party} sur le sujet ou la question suivante, en utilisant uniquement le contexte suivant :

<context>
{context}
</context>                                         

                                                                           
Question: {input}""")

def prompt_no_context(party : str, input : str) -> str : 

    return f"Essaie de répondre en français uniquement à la question suivante en précisant bien que tu n'as pas identifié de réponse à cette question parmi les explications de vote des députés européens de ce parti  {party}. Question: {input}"


document_chain = create_stuff_documents_chain(model, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)




# user_query 
# user_query = st.chat_input("Pose moi une question sur les activités du parti au Parlement")


# if user_query :
#     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
#     st.write(f"Votre sujet : {user_query}")
#     st.write("")
#     st.write(f"{response['answer']}")

messages = st.container()
if user_query := st.chat_input("Pose moi une question sur les activités du parti au Parlement"):
    response = retrieval_chain.invoke({'input': user_query, 'party' : party})
    messages.chat_message("user").write(user_query)

    assistant = assistant_mapper_party[party]

    if len(response["context"])> 0 : 
        messages.chat_message("assistant").write(f"{assistant} : {response['answer']}")
        with messages.chat_message("Jammy", avatar = '🔎') : 
            st.write(f"Jamy : Pour répondre, Valerag a utilisé les explications de vote suivantes ")
            for doc in response["context"] : 
                st.write(f"Explication de vote  de {doc.metadata['name']} ({doc.metadata['orientation']}) sur le sujet : { doc.metadata['source'][:-10]}")
                with st.expander("Voir l'explication") : 
                    st.write(doc.page_content[2:])

    if len(response["context"])==0 : 
        messages.chat_message("assistant").write(f"Valerag : {model.invoke(input = prompt_no_context(input= user_query, party = party)).content}")



# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])


# def generate_response(user_query, party):
#     # Hugging Face Login
#     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
#     # Create ChatBot                        
    
#     return response["answer"]

# if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)