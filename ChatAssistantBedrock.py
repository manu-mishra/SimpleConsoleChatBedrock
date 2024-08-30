#ChatAssistantBedrock.py
import boto3
import json
from botocore.exceptions import ClientError

class ChatAssistantBedrock:
    def __init__(self, model_id, knowledge_base_id, region='us-east-1'):
        """
        Initializes the chat assistant with a model and knowledge base.
        Sets up clients to interact with AWS Bedrock services.
        """
        self.model_id = model_id
        self.knowledge_base_id = knowledge_base_id
        
        # Initialize the AWS Bedrock LLM client and retriever
        self.bedrock_agent_client = boto3.client(
            service_name="bedrock-agent-runtime",
            region_name=region
        )
        self.bedrock_client = boto3.client(
            service_name="bedrock-runtime",
            region_name=region
        )

    def infer(self, prompt):
        """
        Sends a prompt to the Bedrock model and returns the model's response.
        """
        native_request = {
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
        }
        request = json.dumps(native_request)
        try:
            response = self.bedrock_client.invoke_model(modelId=self.model_id, body=request)
            model_response = json.loads(response["body"].read())
            response_text = model_response["outputs"][0]["text"]
            return response_text
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't invoke '{self.model_id}'. Reason: {e}")
            return None

    def retrieve(self, query):
        """
        Retrieves information relevant to the query from the Bedrock knowledge base.
        Returns a list of dictionaries with text content and source URIs.
        """
        try:
            response = self.bedrock_agent_client.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={'text': query},
                retrievalConfiguration={'vectorSearchConfiguration': {'numberOfResults': 2}},
                nextToken='loan'
            )
            results = [
                {
                    'text': result['content']['text'],
                    'score': result['score'],
                    'sourceUri': result['metadata']['x-amz-bedrock-kb-source-uri']
                }
                for result in response.get('retrievalResults', [])
            ]
            return results
        except (ClientError, Exception) as e:
            print(f"ERROR: Can't retrieve from '{self.knowledge_base_id}'. Reason: {e}")
            return None

    def chat(self, query, history):
        """
        Handles the chat interaction given a query and the existing chat history.
        Uses dual infer processes with specific prompts: one for query reformulation and another for response generation.
        """
        formatted_history=""
        if history:  # Check if history is non-empty and not null
            # Step1: First infer process to reformulate the query based on the history
            last_six_messages = history[-6:] if len(history) >= 4 else history[:]
            formatted_history = " ".join(
                f"[INST] {message} [/INST]" for _, message in last_six_messages
            )
            reformulation_prompt = (
                "<<SYS>>"
                "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
                "<</SYS>>"
                f"{formatted_history}."
                f"<s>[INST] {query}[/INST]"
            )
            reformulated_query = self.infer(reformulation_prompt)
        else:
            reformulated_query = query  # Use the original query if history is empty

        # Step2: Retrieve relevant information using the (reformulated) query
        retrieved_information = self.retrieve(reformulated_query)
        contextual_info = ". ".join([info['text'] for info in retrieved_information]) if retrieved_information else "No relevant information found."

        # Step3: infer process to generate the final answer using the retrieved context
        answer_prompt = (
            
            "<<SYS>>"
            "You are resturant customer service bot and your task is to respond to customer's query over chat."
            "Use three sentences maximum and keep the answer concise. "
            "If the user is just greeting, respond with and concise greeting."
            "Use the following information inside <knowledge> tag to answer the question."
            "If you don't know the answer, say that you don't know. "
            "DO NOT mention the word knowledge, context or history in your answer."
            "Do not provide explanations"
            f"<knowledge>{contextual_info}</knowledge>"
            "<</SYS>>"
            #"<s>[INST]Answer the question in direct manner. Only answer the question, nothing else.[/INST]"
            f"{formatted_history}"
            "<s>[INST]"
            f"{query}"
            "[/INST]"
        )
        final_answer = self.infer(answer_prompt)

        return final_answer or "I'm unable to provide an answer at this moment."

