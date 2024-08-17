import streamlit as st
import pandas as pd
import itertools
import random
import re
import os
from PIL import Image

from langchain_anthropic import ChatAnthropic
from langchain_huggingface import HuggingFaceEmbeddings

from agents.vectorstore import Vectorstore
from agents.qa_generation.qa_generation import QAGeneration
from agents.qa_eval_context.qa_eval_context import QAContextEval

from dotenv import load_dotenv
load_dotenv()


def get_vectorestore(pdf_docs, model_name="sentence-transformers/all-MiniLM-L6-v2", store_name='test_vectorstore', save=True):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    vectorstore = Vectorstore(embeddings=embeddings, save=save)
    vecs = vectorstore.generate_vectorstore(pdf_docs, store_name, chunk_size=2500, chunk_overlap=300)
    return vecs

def get_questions(vectorstore, input_text_questions, k=20):
    llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0.3)
    qa_chain = QAGeneration(llm=llm)
    chunks_similarities = vectorstore.similarity_search(input_text_questions, k=k)
    random.shuffle(chunks_similarities)
    qa_generated = []
    for chunk in chunks_similarities:
        try:
            resp = qa_chain.chain.invoke(chunk.page_content)
            text = resp['text']
            question = resp['questions']['question']
            answer = resp['questions']['answer']
            qa_generated.append({'text': text, 'question': question, 'answer': answer})
        except Exception as e:
            print('PROBLEMA AL GENERAR RESPUESTA')
            print(e)
    return qa_generated

def get_grad(text):
    match = re.search(r'CALIFICACIÓN:\s*(\w+)', text)
    print(match)
    if match:
        print('Extract')
        calificacion = match.group(1)
        print(calificacion)
    else:
        calificacion = 'No encontrado'
    print(calificacion)
    return calificacion

@st.cache_data
def processing_question(questions):
    vecs = get_vectorestore(pdf_docs=['./Documents/Analisis-COP3-Escazu_ONG-CEUS-Chile.pdf'],
                            model_name="sentence-transformers/all-MiniLM-L6-v2",
                            store_name='test_vectorstore', save=True)
    input_text_questions = 'acuerdos realizados'
    qa_generated = get_questions(vecs, input_text_questions=input_text_questions, k=questions)
    return qa_generated


llm_eval = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
eval_chain = QAContextEval(llm=llm_eval)
def main():
    st.set_page_config(page_title="Evaluation Demo", page_icon=":closed_books:")

    # logo = Image.open('./assets/logo.png')
    # half = 0.1
    # logo = logo.resize( [int(half * s) for s in logo.size] )
    # st.image(logo)

    st.title("Evaluación del texto")

    number_questions = st.number_input("Ingrese el numero de preguntas que desea: ", min_value=1, max_value=20, value=5)
    qa_generated = processing_question(questions=number_questions)
    if len(qa_generated) > 0:
        for index, question_answer in enumerate(qa_generated):
            question = question_answer["question"]
            context = question_answer["text"]
            st.write(f"*Pregunta {index + 1}:* {question}")
            user_answer = st.text_input(f"Tu respuesta {index+1}:")
            if st.button(f"Corregir Respuesta {index+1}"):
                try:
                    evaluacion = eval_chain.evaluate(query=question, context=context, result=user_answer)
                    try:
                        grade = evaluacion['questions'].content
                        st.write(f"*Pregunta {index + 1}:* {grade}")
                        if grade == 'INCORRECTA':
                            st.write(question_answer['answer'])
                    except Exception as e:
                        print('PROBLEMA EXTRAER REPUESTA')
                        print(e)
                except Exception as e:
                    print('PROBLEMA EVALUAR PREGUNTA')
                    print(e)



if __name__ == "__main__":
    main()