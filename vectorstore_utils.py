from langchain_community.document_loaders import PyPDFLoader
from uuid import uuid4

from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def allowed_file(filename, allowedExtensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowedExtensions


def pdf_to_vectorstore(filepath, fileId, vectorStore):

    loader = PyPDFLoader(filepath)
    
    docs = loader.load()
    ids = [str(uuid4()) for _ in range(len(docs))]

    # For adding metadata to each chunk
    for doc in docs:
        doc.metadata["source_file"] = filepath
        doc.metadata["file_id"] = str(fileId)


    vectorStore.add_documents(documents=docs, ids=ids)

    return len(docs)


def inference(question, retriever):
    
    template ="""Role: You are an expert B2B sales strategist and solutions architect specializing in identifying cross-sell opportunities.
Goal: Based on the retrieved context, analyze the customer’s existing technology environment, company best practices, and vendor product portfolio to recommend additional products, services, or upgrades that align with the customer’s business goals and technology stack.
Context Provided: {context}
Your Tasks:
1.	Identify potential cross-sell recommendations that complement the customer’s existing environment.
2.	For each recommendation, provide:
o   Product / Service Name
o	Why it fits this customer (alignment with environment, needs, or gaps)
o	Business or technical value (efficiency, performance, ROI, etc.)
o	Level of confidence (High / Medium / Low)
3.	Highlight any dependencies, upgrade paths, or pre-requisites if applicable.
4.	Suggest how to position the recommendation during a sales conversation.
Output Format:
### Cross-Sell Recommendations
1. **[Product Name]**
   - **Fit Rationale:** …
   - **Value Proposition:** …
   - **Confidence Level:** …
   - **Sales Positioning Tip:** …

2. **[Product Name]**
   - ...
Constraints:
•	Base all insights only on the provided context and retrieved information.
•	Do not hallucinate unavailable data; if information is missing, state what additional data would improve accuracy.
•	Use concise, professional, and actionable language suitable for sales enablement documentation.
Question: {question}
"""
 
    prompt = ChatPromptTemplate.from_template(template)

    llm = OllamaLLM(model="llama3.1")

    

    augment_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt

    augmented_prompt = augment_chain.invoke(question).to_string()

    generate_chain =  llm | StrOutputParser()
    response = generate_chain.invoke(augmented_prompt)

    return augmented_prompt, response