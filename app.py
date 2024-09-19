from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import UnstructuredFileLoader
from typing import List, Dict, Tuple
import gradio as gr
import validators
import requests
import mimetypes
import tempfile
import os
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts.prompt import PromptTemplate
import pandas as pd
from langchain_experimental.agents.agent_toolkits import create_csv_agent
# from langchain.agents import create_pandas_dataframe_agent
# from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
# from langchain.agents import create_csv_agent
from langchain import OpenAI, LLMChain
from openai import AzureOpenAI

class ChatDocumentQA:
    def __init__(self) -> None:
        pass

    def _get_empty_state(self) -> Dict[str, None]:
        """Create an empty knowledge base."""
        return {"knowledge_base": None}

    def _extract_text_from_pdfs(self, file_paths: List[str]) -> List[str]:
        """Extract text content from PDF files.

        Args:
            file_paths (List[str]): List of file paths.

        Returns:
            List[str]: Extracted text from the PDFs.
        """
        docs = []
        loaders = [UnstructuredFileLoader(file_obj, strategy="fast") for file_obj in file_paths]
        for loader in loaders:
            docs.extend(loader.load())
        return docs

    def _get_content_from_url(self, urls: str) -> List[str]:
        """Fetch content from given URLs.

        Args:
            urls (str): Comma-separated URLs.

        Returns:
            List[str]: List of text content fetched from the URLs.
        """
        file_paths = []
        for url in urls.split(','):
            if validators.url(url):
                headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',}
                r = requests.get(url, headers=headers)
                if r.status_code != 200:
                    raise ValueError("Check the url of your file; returned status code %s" % r.status_code)
                content_type = r.headers.get("content-type")
                file_extension = mimetypes.guess_extension(content_type)
                temp_file = tempfile.NamedTemporaryFile(suffix=file_extension, delete=False)
                temp_file.write(r.content)
                file_paths.append(temp_file.name)

        docs = self._extract_text_from_pdfs(file_paths)
        return docs

    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into smaller chunks.

        Args:
            text (str): Input text to be split.

        Returns:
            List[str]: List of smaller text chunks.
        """
        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=6000, chunk_overlap=0, length_function=len)

        chunks = text_splitter.split_documents(text)

        return chunks
        
    def _create_vector_store_from_text_chunks(self, text_chunks: List[str]) -> FAISS:
        """Create a vector store from text chunks.

        Args:
            text_chunks (List[str]): List of text chunks.

        Returns:
            FAISS: Vector store created from the text chunks.
        """
        embeddings = AzureOpenAIEmbeddings(
                        azure_deployment="text-embedding-3-large",
                    )

        return FAISS.from_documents(documents=text_chunks, embedding=embeddings)


    def _create_conversation_chain(self,vectorstore):

        _template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

        Chat History:  {chat_history}
        Follow Up Input: {question}
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # llm = ChatOpenAI(temperature=0)
        llm=AzureChatOpenAI(azure_deployment = "GPT-3")

        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(),
                                                     condense_question_prompt=CONDENSE_QUESTION_PROMPT,
                                                     memory=memory)

    def _get_documents_knowledge_base(self, file_paths: List[str]) -> Tuple[str, Dict[str, FAISS]]:
        """Build knowledge base from uploaded files.

        Args:
            file_paths (List[str]): List of file paths.

        Returns:
            Tuple[str, Dict]: Tuple containing a status message and the knowledge base.
        """
        file_path = file_paths[0].name
        file_extension = os.path.splitext(file_path)[1]

        if file_extension == '.csv':  
            # agent = self.create_agent(file_path)
            # tools = self.get_agent_tools(agent)
            # memory,tools,prompt = self.create_memory_for_csv_qa(tools)
            # agent_chain = self.create_agent_chain_for_csv_qa(memory,tools,prompt)
            agent_chain = create_csv_agent(
                          AzureChatOpenAI(azure_deployment = "GPT-3"),
                          file_path,
                          verbose=True,
                          agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            )
            return "file uploaded", {"knowledge_base": agent_chain}
            
        else:
            pdf_docs = [file_path.name for file_path in file_paths]
            raw_text = self._extract_text_from_pdfs(pdf_docs)
            text_chunks = self._split_text_into_chunks(raw_text)
            vectorstore = self._create_vector_store_from_text_chunks(text_chunks)
            return "file uploaded", {"knowledge_base": vectorstore}        


    def _get_urls_knowledge_base(self, urls: str) -> Tuple[str, Dict[str, FAISS]]:
        """Build knowledge base from URLs.

        Args:
            urls (str): Comma-separated URLs.

        Returns:
            Tuple[str, Dict]: Tuple containing a status message and the knowledge base.
        """
        webpage_text = self._get_content_from_url(urls)
        text_chunks = self._split_text_into_chunks(webpage_text)
        vectorstore = self._create_vector_store_from_text_chunks(text_chunks)
        return "file uploaded", {"knowledge_base": vectorstore}

#************************
#   csv qa
#************************
    def create_agent(self,file_path):
        agent_chain = create_csv_agent(
        AzureChatOpenAI(azure_deployment = "GPT-3"),
        file_path,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )
        return agent_chain
    def get_agent_tools(self,agent):
      # search = agent
      tools = [
        Tool(
            name="dataframe qa",
            func=agent.run,
            description="useful for when you need to answer questions about table data and dataframe data",
        )
      ]
      return tools

    def create_memory_for_csv_qa(self,tools):
      prefix = """Have a conversation with a human, answering the following questions about table data and dataframe data as best you can. You have access to the following tools:"""
      suffix = """Begin!"

      {chat_history}
      Question: {input}
      {agent_scratchpad}"""

      prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
      )
      memory = ConversationBufferMemory(memory_key="chat_history",return_messages=True)

      return memory,tools,prompt

    def create_agent_chain_for_csv_qa(self,memory,tools,prompt):

        llm_chain = LLMChain(llm=AzureChatOpenAI(azure_deployment = "GPT-3"), prompt=prompt)
        agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        agent_chain = AgentExecutor.from_agent_and_tools(
            agent=agent, tools=tools, verbose=True, memory=memory
        )

        return agent_chain

    def _get_response(self, message: str, chat_history: List[Tuple[str, str]], state: Dict[str, FAISS],file_paths) -> Tuple[str, List[Tuple[str, str]]]:
        """Get a response from the chatbot.

        Args:
            message (str): User's message/question.
            chat_history (List[Tuple[str, str]]): List of chat history as tuples of (user_message, bot_response).
            state (dict): State containing the knowledge base.

        Returns:
            Tuple[str, List[Tuple[str, str]]]: Tuple containing a status message and updated chat history.
        """
        try:
          if file_paths:
            file_path = file_paths[0].name
            file_extension = os.path.splitext(file_path)[1]

            if file_extension == '.csv':
                agent_chain = state["knowledge_base"]
                response = agent_chain.run(input = message)
                chat_history.append((message, response))
                return "", chat_history
            
            else:
                vectorstore = state["knowledge_base"]
                chat = self._create_conversation_chain(vectorstore)
                print("chat_history",chat_history)
                response = chat({"question": message,"chat_history": chat_history})
                chat_history.append((message, response["answer"]))
                return "", chat_history
        except:
            chat_history.append((message, "Please Upload Document or URL"))
            return "", chat_history

    def gradio_interface(self) -> None:
        """Create a Gradio interface for the chatbot."""
        with gr.Blocks(css = "style.css" ,theme="freddyaboulton/test-blue") as demo:
            
            gr.HTML("""<center class="darkblue" text-align:center;padding:30px;'><center>
            <center><h1 class ="center" style="color:#fff"></h1></center>
            <br><center><h1 style="color:#fff">Virtual Assistant Chatbot</h1></center>""")     
            
            state = gr.State(self._get_empty_state())
            chatbot = gr.Chatbot()
            
            with gr.Row():
                with gr.Column(scale=0.85):
                    msg = gr.Textbox(label="Question")
                with gr.Column(scale=0.15):
                    file_output = gr.Textbox(label="File Status")
            with gr.Row():
                with gr.Column(scale=0.85):
                    clear = gr.ClearButton([msg, chatbot])
                with gr.Column(scale=0.15):
                    upload_button = gr.UploadButton(
                        "Browse File",
                        file_types=[".txt", ".pdf", ".doc", ".docx", ".csv"],
                        file_count="multiple", variant="primary"
                    )
            with gr.Row():
                with gr.Column(scale=1):
                    input_url = gr.Textbox(label="urls")

            input_url.submit(self._get_urls_knowledge_base, input_url, [file_output, state])
            upload_button.upload(self._get_documents_knowledge_base, upload_button, [file_output, state])
            msg.submit(self._get_response, [msg, chatbot, state,upload_button], [msg, chatbot])

        demo.launch()


if __name__ == "__main__":
    chatdocumentqa = ChatDocumentQA()
    chatdocumentqa.gradio_interface()