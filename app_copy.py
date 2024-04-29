# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
#import chromadb

import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import qdrant_client

from langchain.document_loaders import CSVLoader
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
#from langchain_core.st import HumanMessage , AIMessage
#from langchain_chroma import Chroma
from langchain_community.vectorstores import Qdrant
import os


from rag.utils import mapper_partis, assistant_mapper_party, get_response, stream_str




#Setting tokens
api_key = st.secrets["API_MISTRAL"]
hugging_face_token = st.secrets["HUGGING_FACE"]
api_qdrant = st.secrets["API_QDRANT"]
url_qdrant = st.secrets["URL_CLUSTER_QDRANT"]
from huggingface_hub import login
login(token=hugging_face_token)


client = qdrant_client.QdrantClient(
    url_qdrant,
    api_key=api_qdrant, # For Qdrant Cloud, None for local instance
)



#Setting style of streamlit page 
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

# Remove whitespace from the top of the page and sidebar
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)



#Tools for the rag

embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
model = ChatMistralAI(mistral_api_key=api_key,
                      temperature=.2, 
                      stream = True, 
                      safe_prompt = True)


#Loading vector db from qdrant
db = Qdrant(
    client=client, 
    collection_name="my_documents", 
    embeddings=embeddings)




st.warning('Version demo : nous ne saurions être tenus responsables si votre député(e) préféré(e) est diffamé(e)', icon="⚠️")

#Prompt for common use
prompt = ChatPromptTemplate.from_template("""Résume, en français uniquement, la position des députés européens du parti {party} sur le sujet ou la question suivante, en utilisant uniquement le contexte suivant :

<context>
{context}
</context>                                         

                                                                           
Question: {input}""")

#Prompt if no context can be retrieved
def prompt_no_context(party : str, input : str) -> str : 

    return f"Essaie de répondre en français uniquement à la question suivante en précisant bien que tu n'as pas identifié de réponse à cette question parmi les explications de vote des députés européens de ce parti  {party}. Question: {input}"







#messages = st.container(height=500)
with st.chat_message("assistant"):
    st.write(f"Bonjour ! Je suis marIAnne. Passionnée par la politique, républicaine, et persuadée que la démocratie passe par la transparence, j'ai décidé de lire toutes les explications de vote des députés européens français lors des dernières années. Alors, pour savoir ce qu'ont pensé et surtout, voté, les députés qui t'intéressent, pose moi tes questions. Ainsi, tu peux te faire ton avis en t'appuyant sur les actes plutôt que sur les discours ! ")
    st.write("Choisis un parti politique qui t'intéresse. Ecris une question ou un sujet, et je te résumerai les prises de positions des députés européens dessus.")
    party = mapper_partis[st.selectbox(label = "Quel est le groupe politique dont tu souhaites connaître les positions prises ? " , options=mapper_partis.keys())]

    retriever = db.as_retriever(
    search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, 'filter' :  {"party": party}}
    )

    document_chain = create_stuff_documents_chain(model, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    

if user_query := st.chat_input("Pose moi une question sur les activités du parti au Parlement"):
    
    #response = retrieval_chain.stream({'input': user_query, 'party' : party})
    st.chat_message("user").write(user_query)
    context = retriever.invoke(user_query)
    
    if len(context)> 0 :
        with st.chat_message("assistant") : 
            st.write_stream(get_response(retrieval_chain, user_input=user_query, party = party))
        with st.chat_message("Jammy", avatar = '🔎') : 
            st.write_stream(stream_str("Jamy : Pour répondre, Valerag a utilisé les explications de vote suivantes "))
            for doc in context : 
                st.write_stream(stream_str(f"Explication de vote  de {doc.metadata['name']} ({doc.metadata['orientation']}) sur le sujet : { doc.metadata['source'][:-10]}"))
                with st.expander("Voir l'explication") : 
                    st.write(doc.page_content[2:])

    if len(context)==0 : 
        st.chat_message("assistant").write_stream(f"MarIAnne : {model.invoke(input = prompt_no_context(input= user_query, party = party)).content}")



# if "st" not in st.session_state.keys():
#     st.session_state.st = [{"role": "assistant", "content": "How may I help you?"}]

# for message in st.session_state.st:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])


# def generate_response(user_query, party):
#     # Hugging Face Login
#     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
#     # Create ChatBot                        
    
#     return response["answer"]

# if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
#     st.session_state.st.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#     st.write(prompt)




# # __import__('pysqlite3')
# # import sys
# # sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# #import chromadb
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()
# import qdrant_client

# from langchain.document_loaders import CSVLoader
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_mistralai.embeddings import MistralAIEmbeddings
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain.chains import create_retrieval_chain
# #from langchain_chroma import Chroma


# from langchain_community.vectorstores import Qdrant

# import os


# from rag.utils import mapper_partis, assistant_mapper_party

# #from st_files_connection import FilesConnection

# # Create connection object and retrieve file contents.
# # Specify input format is a csv and to cache the result for 600 seconds.
# # conn = st.connection('s3', type=FilesConnection)
# # df = conn.read("testbucket-jrieke/myfile.csv", input_format="csv", ttl=600)



# #load_dotenv()
# api_key = st.secrets["API_MISTRAL"]
# hugging_face_token = st.secrets["HUGGING_FACE"]
# api_qdrant = st.secrets["API_QDRANT"]
# url_qdrant = st.secrets["URL_CLUSTER_QDRANT"]
# from huggingface_hub import login
# login(token=hugging_face_token)


# client = qdrant_client.QdrantClient(
#     url_qdrant,
#     api_key=api_qdrant, # For Qdrant Cloud, None for local instance
# )

# #os.getenv("API_MISTRAL")


# streamlit_style = """
# 			<style>
# 			html, body, [class*="css"]  {
# 			font-family: 'Impact', sans-serif;
# 			}
# 			</style>
# 			"""
# st.markdown(streamlit_style, unsafe_allow_html=True)


# st.markdown("""
# <style>
# .big-font {
#     font-size:50px !important;
#     font-family:Impact;
# }
            
# .medium-font {
#     font-size:30px ;
#     font-family:Impact;
# }
# </style>
# """, unsafe_allow_html=True)
# st.markdown('<p class="big-font"> DemocracIA </p>', unsafe_allow_html=True)
# st.markdown('<p class="medium-font"> Nous votons, eux aussi 👀  </p>', unsafe_allow_html=True)




# # Afficher le titre avec la police personnalisée
# #st.title('ValeRAG Hayer')

# embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
# model = ChatMistralAI(mistral_api_key=api_key,
#                       temperature=.2, 
#                       stream = True, 
#                       safe_prompt = True)



# db = Qdrant(
#     client=client, 
#     collection_name="my_documents", 
#     embeddings=embeddings)

# #def load_data() : 
#     # with st.spinner(text="Chargement de la base de données, patience !"):
#     #     loader = CSVLoader('db/all_french_explanations.csv',metadata_columns=["party", "number","name", "source", "source_date","orientation"], encoding="utf-8")
#     #     documents =loader.load()
#     #     db = Chroma.from_documents(documents[:20000], embeddings, persist_directory="../chroma_db")
#     #     return db


# #db = load_data()





# #db = Chroma(persist_directory="chroma_db", embedding_function=embeddings)




# st.warning('Version demo : nous ne saurions être tenus responsables si votre député(e) préféré(e) est diffamé(e)', icon="⚠️")

# party = mapper_partis[st.selectbox(label = "Parti politique" , options=mapper_partis.keys())]

# retriever = db.as_retriever(
#     search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.6, 'filter' :  {"party": party}}
# )

# # prompt = ChatPromptTemplate.from_template("""Resume only in french language the political position of the MPs of the party {party} on the following subject based only on the provided context:

# # <context>
# # {context}
# # </context>

# # Question: {input}""")

# prompt = ChatPromptTemplate.from_template("""Résume, en français uniquement, la position des députés européens du parti {party} sur le sujet ou la question suivante, en utilisant uniquement le contexte suivant :

# <context>
# {context}
# </context>                                         

                                                                           
# Question: {input}""")

# def prompt_no_context(party : str, input : str) -> str : 

#     return f"Essaie de répondre en français uniquement à la question suivante en précisant bien que tu n'as pas identifié de réponse à cette question parmi les explications de vote des députés européens de ce parti  {party}. Question: {input}"


# document_chain = create_stuff_documents_chain(model, prompt)
# retrieval_chain = create_retrieval_chain(retriever, document_chain)




# # user_query 
# # user_query = st.chat_input("Pose moi une question sur les activités du parti au Parlement")


# # if user_query :
# #     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
# #     st.write(f"Votre sujet : {user_query}")
# #     st.write("")
# #     st.write(f"{response['answer']}")

# messages = st.container()
# if user_query := st.chat_input("Pose moi une question sur les activités du parti au Parlement"):
#     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
#     messages.chat_message("user").write(user_query)

#     assistant = assistant_mapper_party[party]

#     if len(response["context"])> 0 : 
#         messages.chat_message("assistant").write(f"{assistant} : {response['answer']}")
#         with messages.chat_message("Jammy", avatar = '🔎') : 
#             st.write(f"Jamy : Pour répondre, Valerag a utilisé les explications de vote suivantes ")
#             for doc in response["context"] : 
#                 st.write(f"Explication de vote  de {doc.metadata['name']} ({doc.metadata['orientation']}) sur le sujet : { doc.metadata['source'][:-10]}")
#                 with st.expander("Voir l'explication") : 
#                     st.write(doc.page_content[2:])

#     if len(response["context"])==0 : 
#         messages.chat_message("assistant").write(f"Valerag : {model.invoke(input = prompt_no_context(input= user_query, party = party)).content}")



# # if "messages" not in st.session_state.keys():
# #     st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

# # for message in st.session_state.messages:
# #     with st.chat_message(message["role"]):
# #         st.write(message["content"])


# # def generate_response(user_query, party):
# #     # Hugging Face Login
# #     response = retrieval_chain.invoke({'input': user_query, 'party' : party})
# #     # Create ChatBot                        
    
# #     return response["answer"]

# # if prompt := st.chat_input(disabled=not (hf_email and hf_pass)):
# #     st.session_state.messages.append({"role": "user", "content": prompt})
# #     with st.chat_message("user"):
# #         st.write(prompt)