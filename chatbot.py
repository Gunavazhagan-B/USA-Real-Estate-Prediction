from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class RealEstateChatbot:
    def __init__(self):
        self.template="""
        Answer the question below:

        Here is the conversation history : {context}

        Question : {question}

        Answer :
        """

        self.model=OllamaLLM(model='llama3')
        self.prompt=ChatPromptTemplate.from_template(template=self.template)
        self.chain=self.prompt|self.model
        self.context=""

    def ask(self,question: str)->str:
        response=self.chain.invoke({'context':self.context,'question':question})
        self.context+=f"\nUser: {question}\nAI: {response}"
        return response




