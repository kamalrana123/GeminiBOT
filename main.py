import os
import google.generativeai as genai
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI



load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def readPdf():
    pdf_path = './pdf/poshpolicy214202314455809.pdf'
    extracted_text = PyPDFLoader(pdf_path)
    pages = extracted_text.load_and_split()
    return pages


def textSplitter(pages):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    context = "\n\n".join(str(p.page_content) for p in pages)
    print(len(pages))
    texts = text_splitter.split_text(context)
    return texts


def createEmbeddingObject():
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    print("embedding object created", embedding)
    return embedding


def createVectorIndex(texts,embedding):
    vector_index = Chroma.from_texts(texts,embedding).as_retriever()
    return vector_index

def getRelevantDocuments(question,vector_index):
    docs = vector_index.get_relevant_documents(question)
    return docs



def createPrompt():
    prompt_template =  """ANswer the question as precise as possible using the provided context. If the answer is not contained in the context, say "answer not available in context"\n\n
                          context: \n {context}?\n
                          question:\n {question} \n
                          answer:"""
    return prompt_template

def load_question_answer_chain(prompt,pages,question):
    model = ChatGoogleGenerativeAI(google_api_key=GOOGLE_API_KEY,model="gemini-pro", temperature=0.5)
    prompt_template = createPrompt()
    prompt = PromptTemplate(
        template = prompt_template, input_variables= ["context", "question"]
    )
    stuff_chain = load_qa_chain(model,chain_type='stuff',prompt=prompt)
    stuff_answer = stuff_chain(
        {"input_documents":pages, "question":question},return_only_outputs=True
        )
    aimodel = genai.GenerativeModel('gemini-pro')
    print(stuff_answer)
    result = aimodel.generate_content([question,stuff_answer['output_text']])
    print(result.text)


if __name__ == "__main__":
    pages = readPdf()
    texts = textSplitter(pages)
    embedding = createEmbeddingObject()
    vector_index = createVectorIndex(texts,embedding)
    prompt_template = createPrompt()
    question = "how to file a complaint? explain step wise"
    docs = getRelevantDocuments(question,vector_index)
    load_question_answer_chain(prompt_template,docs,question)
