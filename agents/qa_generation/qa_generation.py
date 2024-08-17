from agents.qa_generation.prompts import CHAT_PROMPT as prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.base import RunnableEach


class QAGeneration:
    def __init__(self, llm, prompt=prompt):
        self.llm = llm
        self.prompt = prompt
        chain = RunnableParallel(
            text=RunnablePassthrough(),
            questions=(
                self.prompt | self.llm | JsonOutputParser()
            )
        )
        self.chain = chain



