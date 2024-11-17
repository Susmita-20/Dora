import streamlit as st
import os
from menu import menu
import pandas as pd
from lida import Manager, TextGenerationConfig, llm
import io
from PIL import Image
from io import BytesIO
import base64
import matplotlib.pyplot as plt
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.experimental.query_engine.pandas import (
    PandasInstructionParser,
)
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate


st.set_page_config(page_title="DORA", page_icon="🦙")
menu()


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

instruction_str = (
    "1. Convert the query to executable Python code using Pandas.\n"
    "2. The final line of code should be a Python expression that can be called with the `eval()` function.\n"
    "3. The code should represent a solution to the query.\n"
    "4. PRINT ONLY THE EXPRESSION.\n"
    "5. Do not quote the expression.\n"
)

pandas_prompt_str = (
    "You are working with a pandas dataframe in Python.\n"
    "The name of the dataframe is `df`.\n"
    "This is the result of `print(df.head())`:\n"
    "{df_str}\n\n"
    "Follow these instructions:\n"
    "{instruction_str}\n"
    "Query: {query_str}\n\n"
    "Expression:"
)
response_synthesis_prompt_str = (
    "Given an input question, synthesize a response from the query results.\n"
    "Query: {query_str}\n\n"
    "Pandas Instructions (optional):\n{pandas_instructions}\n\n"
    "Pandas Output: {pandas_output}\n\n"
    "Response: "
)

def base64_to_image(base64_string):
	image = Image.open(BytesIO(base64.b64decode(base64_string)))
	return image

lida = Manager(text_gen = llm("openai"))
text_gen_config = TextGenerationConfig(n=1,temperature=0.1,model="gpt-4o-mini",use_cache=False)

try: 
    
    project_name = st.sidebar.selectbox("Select Project:", options=st.session_state.projects)
    st.sidebar.write(f"Selected Project: {project_name}")
    if os.path.exists(f"{st.session_state.role}/{project_name}"):
        files = os.listdir(f"{st.session_state.role}/{project_name}")
        if files:
            for file in files:
                if 'csv' in file.split('.'):
                    st.sidebar.write(file)
                else:
                    st.sidebar.write("No CSV files found.")
        else:
            st.sidebar.write("The project is empty.")

except Exception as e:
    st.write("No Dataset Found.")
    st.stop()
llm = OpenAI(model="gpt-4o-mini")
def visualize():
    query = st.chat_input(f"Enter Query:")
    if query:
        if ('plot' in query or 'graph' in query or 'chart' in query or 'visual' in query or 'visualization' in query):
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message(f"user"):
                    st.markdown(query)
            for file in files:
                if 'csv' in file.split('.'):
                    filename = file
            # df = pd.read_csv(f"{st.session_state.role}/{project_name}/{filename}")
            # query = "how many people got saved in the titanic disaster?"
            summary = lida.summarize(f"{st.session_state.role}/{project_name}/{filename}",summary_method = "default",textgen_config = text_gen_config)
            
        else:
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message(f"user"):
                    st.markdown(query)
            for file in files:
                if 'csv' in file.split('.'):
                    filename = file
            df = pd.read_csv(f"{st.session_state.role}/{project_name}/{filename}")
            pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
                instruction_str=instruction_str, df_str=df.head(5)
            )
            pandas_output_parser = PandasInstructionParser(df)
            response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)

            qp = QP(
                modules={
                    "input": InputComponent(),
                    "pandas_prompt": pandas_prompt,
                    "llm1": llm,
                    "pandas_output_parser": pandas_output_parser,
                    "response_synthesis_prompt": response_synthesis_prompt,
                    "llm2": llm,
                },
                verbose=True,
            )
            qp.add_chain(["input", "pandas_prompt", "llm1", "pandas_output_parser"])
            qp.add_links(
                [
                    Link("input", "response_synthesis_prompt", dest_key="query_str"),
                    Link(
                        "llm1", "response_synthesis_prompt", dest_key="pandas_instructions"
                    ),
                    Link(
                        "pandas_output_parser",
                        "response_synthesis_prompt",
                        dest_key="pandas_output",
                    ),
                ]
            )
            # add link from response synthesis prompt to llm2
            qp.add_link("response_synthesis_prompt", "llm2")
        
        # response = chat_engine.chat(query)
        with st.chat_message("assistant"):
            with st.spinner("Grabbing the answers..."):
                if "plot" in query or "graph" in query or "chart" in query or "visual" in query or "visualization" in query:
                    charts = lida.visualize(summary = summary,goal = query,textgen_config = text_gen_config)
                    image = base64_to_image(charts[0].raster)
                    # save the image
                    # image.save("output.png")
                    # st.pyplot(response.response)
                    st.session_state.messages.append({"role": "assistant", "content": st.image(image=image, caption='Visualization')})
                else:
                    response = qp.run(
                                query_str=query,
                            )
                    st.markdown(response.message.content)
                    st.session_state.messages.append({"role": "assistant", "content": response.message.content})
                # st.markdown(st.session_state.messages)

if __name__ == "__main__":
    visualize()