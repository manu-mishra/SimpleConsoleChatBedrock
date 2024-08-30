from ChatAssistantLangChain import ChatAssistantLangChain
from ChatAssistantBedrock import ChatAssistantBedrock
import colorama
from colorama import Fore, Style
colorama.init(autoreset=True)

def main():
    model_id="mistral.mixtral-8x7b-instruct-v0:1"
    knowledge_base_id="YOUR_KNOWLEDGE_BASE_ID"
    # Initialize the ChatAssistant with specific IDs

    # Uncomment following lines to 
    # Compare between LangChain vs Plain Bedrock implementation
   
   #assistant = ChatAssistantLangChain(model_id=model_id, knowledge_base_id=knowledge_base_id)
    assistant = ChatAssistantBedrock(model_id=model_id, knowledge_base_id=knowledge_base_id)
    history = []  # Initialize empty history

    while True:
        user_input = input(f"{Fore.RED}Enter your query (type 'exit' to stop): {Style.RESET_ALL}")
        if user_input.lower() == 'exit':
            break
        response = assistant.chat(user_input, history)
        
        # Update history with labeled interactions
        history.extend([
            ("human", user_input),  
            ("assistant", response) 
        ])

        print(f"{Fore.GREEN}Answer: {response}{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
