import streamlit as st
import os
from menu import menu
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex, StorageContext, load_index_from_storage
from llama_index.core.memory import ChatMemoryBuffer

st.set_page_config(page_title="DORA", page_icon="🦙")
menu()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

gen_prompt = "Leverage your chatbot abilities to answer some given questions on a specific topic by only using the context provided, not using any prior knowledge, making sure to avoid repetitions in the informations and write the answers in such a way that all the answers must follow the flow and together can be used to form a report."

try: 
    projects_names = os.listdir(f"{st.session_state.role}/index")
    project_name = st.sidebar.selectbox("Select Project:", options=projects_names)
    st.sidebar.write(f"Selected Project: {project_name}")
    if os.path.exists(f'{st.session_state.role}/index/{project_name}'):
        if os.path.exists(f"{st.session_state.role}/{project_name}"):
            files = os.listdir(f"{st.session_state.role}/{project_name}")
            if files:
                for file in files:
                    st.sidebar.write(file)
            else:
                st.sidebar.write("The project is empty.")
    else:
        docs = SimpleDirectoryReader(f"{st.session_state.role}/{project_name}").load_data()
        index = VectorStoreIndex.from_documents(docs)
        summary = SummaryIndex.from_documents(docs)
        index.storage_context.persist(f'{st.session_state.role}/index/{project_name}')
        summary.storage_context.persist(f'{st.session_state.role}/summary/{project_name}')

except Exception as e:
    st.write("No Index Found.")
    st.stop()

def query():
    query = st.chat_input(f"Enter Query:")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message(f"user"):
                st.markdown(query)
        if "summary" in query or "short note" in query or "TLDR" in query or "tl;dr" in query:
            storage_context = StorageContext.from_defaults(persist_dir=f'{st.session_state.role}/summary/{project_name}')
            index = load_index_from_storage(storage_context)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=f'{st.session_state.role}/index/{project_name}')
            index = load_index_from_storage(storage_context)
        chat_engine = index.as_chat_engine(
            chat_mode="openai",
            memory=ChatMemoryBuffer.from_defaults(token_limit=500000),
            system_prompt=(
                f"{gen_prompt}\n"
                "Only use the given context, do not add any prior knowledge."
                "the given context : {context_str}"
            ),
            )
        # response = chat_engine.chat(query)
        with st.chat_message("assistant"):
            with st.spinner("Grabbing the answers..."):
                response = chat_engine.chat(query)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
                # st.markdown(st.session_state.messages)



if __name__ == "__main__":
    query()