#ChatAssistantLangChain.py
from langchain_aws import ChatBedrock
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_aws.retrievers import AmazonKnowledgeBasesRetriever
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

class ChatAssistantLangChain:
    def __init__(self, model_id, knowledge_base_id):
        # Initialize model and retriever
        self.llm = ChatBedrock(model_id=model_id, model_kwargs=dict(temperature=0))
        self.retriever = AmazonKnowledgeBasesRetriever(
            knowledge_base_id=knowledge_base_id,
            retrieval_config={"vectorSearchConfiguration": {"numberOfResults": 4}},
        )

        # Setup the final retrieval and answer chain
        self.setup_final_chain()

    def setup_final_chain(self):
        # Contextualization prompt template for history-aware retrieval
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        # Question-Answering system prompt
        system_prompt = (
            "You are resturant customer service bot and your task is to respond to customer's query over chat."
            "Use three sentences maximum and keep the answer concise. "
            "If the user is just greeting, respond with and concise greeting."
            "Use the following information inside <knowledge> tag to answer the question."
            "If you don't know the answer, say that you don't know. "
            "DO NOT mention the word knowledge, context or history in your answer."
            "Do not provide explanations""You are resturant customer service bot and your task is to respond to customer's query over chat."
            "Use three sentences maximum and keep the answer concise. "
            "If the user is just greeting, respond with and concise greeting."
            "Use the following information inside <knowledge> tag to answer the question."
            "If you don't know the answer, say that you don't know. "
            "DO NOT mention the word knowledge, context or history in your answer."
            "Do not provide explanations"
            "<knowledge>{context}</knowledge>"
        )
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        # Create the final retrieval and question-answering chain
        self.rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def chat(self, query, history):
        """Invoke the final RAG chain with the query and chat history, then generate a response."""
        # Convert history to list of AIMessage and HumanMessage
        formatted_history = [HumanMessage(content=msg) if speaker == 'human' else AIMessage(content=msg) for speaker, msg in history]

        # Build the prompt using current query and chat history
        full_prompt = {
            "chat_history": formatted_history,
            "input": query
        }

        # Invoke the final RAG chain with the full prompt
        response = self.rag_chain.invoke(full_prompt)
        return response["answer"]

