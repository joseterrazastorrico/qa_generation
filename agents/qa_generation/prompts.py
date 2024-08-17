# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.prompt import PromptTemplate

templ1 = """Eres un asistente inteligente diseñado para ayudar a los evaluadores a plantear preguntas relacionadas al texto proporcionado.
Dado un fragmento de texto, debe generar una pregunta y  una respuesta que pueda usarse para evaluar las habilidades de comprensión lectora de un estudiante.
Cuando se le ocurra este par de preguntas y respuestas, debe responder en el siguiente formato:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```
Todo lo que esté entre ``` debe ser un json válido.
"""
templ2 = """Cree un par de preguntas y respuestas, en el formato JSON especificado, para el siguiente texto:
----------------
{text}"""
CHAT_PROMPT = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(templ1),
        HumanMessagePromptTemplate.from_template(templ2),
    ]
)
templ = """Eres un asistente inteligente diseñado para ayudar a los evaluadores a plantear preguntas relacionadas al texto proporcionado.
Dado un fragmento de texto, debe generar una pregunta y  una respuesta que pueda usarse para evaluar las habilidades de comprensión lectora de un estudiante.
Cuando se le ocurra este par de preguntas y respuestas, debe responder en el siguiente formato:
```
{{
    "question": "$YOUR_QUESTION_HERE",
    "answer": "$THE_ANSWER_HERE"
}}
```
Todo lo que esté entre ``` debe ser un json válido.

Cree un par de preguntas y respuestas, en el formato JSON especificado, para el siguiente texto:
----------------
{text}"""
PROMPT = PromptTemplate.from_template(templ)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)