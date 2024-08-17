from agents.qa_eval_context.prompts import CONTEXT_PROMPT as prompt
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

class QAContextEval:
    def __init__(self, llm, prompt=prompt):
        self.llm = llm
        self.prompt = prompt
        chain = RunnableParallel(
            text=RunnablePassthrough(),
            questions=(
                self.llm
            )
        )
        self.chain = chain

    def evaluate(self, query, context, result):
        filled_prompt = self.prompt.format(query=query, context=context, result=result)
        result = self.chain.invoke(filled_prompt)
        
        return result



