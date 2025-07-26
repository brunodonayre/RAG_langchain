#!/usr/bin/env python
# coding: utf-8

# # LangChain Quickstart - Create Your Own RAG Chains
# 
# Link: https://www.langchain.com/
# 
# 

# # 1.0 Install the required packages
# 
# Some packages have changed inside Langchain. This is how you should install and import them from now on. Remember that langchain has been divided into three main packages:
# - `langchain` Chains, agents, and retrieval strategies that make up an application's cognitive architecture.
# - Integration packages (e.g. langchain-openai, langchain-anthropic, etc.): Important integrations have been split into lightweight packages that are co-maintained by the LangChain team and the integration developers.
# - `langchain-core` contains the main abstractions of the library.
# - `langchain-community` contains third-party integrations like Chroma, FAISS, etc.
# - Some third-party integrations have their own package (outside `langchain-community`) that you should install if you want to use them. For example, OpenAI has its own package: `langchain-openai`.
# 
# All three are installed when you run `pip install langchain`. But you can install community or core individually as well.
# 
# 
# he LangChain framework consists of multiple open-source libraries. Read more in the Architecture page.
# 
#     

# In[1]:


get_ipython().system(' pip install langchain')
get_ipython().system(' pip install langchain-openai')
get_ipython().system(' pip install langchain-community')


# In[19]:


get_ipython().system('pip show langchain')


# In[2]:


import os
os.environ["OPENAI_API_KEY"] = "TOKEN"


# In[3]:


from langchain_openai import ChatOpenAI

llm = ChatOpenAI()


# In[4]:


llm.invoke("¿Que opciones tiene la funcion ChatOpenAI de langchain_openai?")


# # 2.0 Create your first chain with LCEL (LangChain Expression Language)
# 
# LCEL is now the default way to create chains in LangChain. It has a more pipeline-like syntax and allows you to modify already-existing chains.

# In[5]:


from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un experto en IA que sabe de machine learning y va responder en español"),
    ("user", "{input}")
])


# In[6]:


chain = prompt | llm


# In[7]:


chain.invoke({
    "input": "¿Cual es la diferencia entre aprendizaje supervisado y no supervisado?"
    })


# In[8]:


# add output parser to the chain

from langchain_core.output_parsers import StrOutputParser

output_parser = StrOutputParser()


# In[9]:


chain = prompt | llm | output_parser


# In[10]:


chain.invoke({"input": "¿que es el aprendizaje por refuerzo?"})


# In[11]:


llm.invoke("¿que es el aprendizaje por refuerzo?")


# In[38]:


from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_template("Dime un chiste sobre {topic}")
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Without parser
raw_chain = prompt | llm
response = raw_chain.invoke({"topic": "programación"})
print(type(response))  # <class 'langchain_core.messages.ai.AIMessage'>

# With parser
clean_chain = prompt | llm | StrOutputParser()
clean_response = clean_chain.invoke({"topic": "programación"})
print(type(clean_response))  # <class 'str'>


# # 3.0 Create a Retrieval Chain

# 
# ## 3.1 Load the source documents
# 
# First, we will have to load the documents that will enrich our LLM prompt. We will use [this blog post](https://blog.langchain.dev/langchain-v0-1-0/) from LangChain's official website explaining the new release. OpenAI's models were not trained on this content, so the only way to ask questions about it is to build a RAG chain.
# 
# The first thing to do is to load the blog content to our vector store. We will use beautiful soup to scrap the blog post. Then we will store it in a FAISS vector store.

# In[12]:


# retrieval chain

get_ipython().system(' pip install beautifulsoup4')


# In[13]:


get_ipython().system(' pip install faiss-cpu')


# In[14]:


from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://blog.langchain.dev/langchain-v0-1-0/")

docs = loader.load()


# In[15]:


docs


# In[16]:


from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()


# In[17]:


from langchain_community.vectorstores import FAISS
from langchain.text_splitter  import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)


# In[18]:


documents[1]


# In[20]:


vectorstore = FAISS.from_documents(documents, embeddings)


# ## 3.2 Create a Context-Aware LLM Chain
# 
# Here we create a chain that will answer a question given a context. For now, we are passing the context manually, but in the next step we will pass in the documents fetched from the vector store we created above 👆

# In[21]:


# create chain for documents

from langchain.chains.combine_documents import create_stuff_documents_chain

template = """"Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}
"""
prompt = ChatPromptTemplate.from_template(template)
document_chain = create_stuff_documents_chain(llm, prompt)


# In[22]:


from langchain_core.documents import Document

document_chain.invoke({
    "input": "How many integrations has the langchain 0.1.0?",
    "context": [Document(page_content="Today we‚Äôre excited to announce the release of langchain 0.1.0, our first stable version. It is fully backwards compatible, comes in both Python and JavaScript, and comes with improved focus through both functionality and documentation. A stable version of LangChain helps us earn developer trust and gives us the ability to evolve the library systematically and safely.Python GitHub DiscussionPython v0.1.0 GuidesJS v0.1.0 GuidesYouTube WalkthroughIntroductionLangChain has been around for a little over a year and has changed a lot as it‚Äôs grown to become the default framework for building LLM applications. As we previewed a month ago, we recently decided to make significant changes to the\xa0 LangChain package architecture in order to better organize the project and strengthen the foundation.\xa0Specifically we made two large architectural changes: separating out langchain-core and separating out partner packages (either into langchain-community or standalone partner packages) from langchain.\xa0As a reminder, langchain-core contains the main abstractions, interfaces, and core functionality. This code is stable and has been following a stricter versioning policy for a little over a month now.langchain itself, however, still remained on 0.0.x versions. Having all releases on minor version 0 created a few challenges:Users couldn‚Äôt be confident that updating would not have breaking changeslangchain became bloated and unstable as we took a ‚Äúmaintain everything‚Äù approach to reduce breaking changes and deprecation notificationsHowever, starting today with the release of langchain 0.1.0, all future releases will follow a new versioning standard. Specifically:Any breaking changes to the public API will result in a minor version bump (the second digit)Any bug fixes or new features will result in a patch version bump (the third digit)We hope that this, combined with the previous architectural changes, will:Communicate clearly if breaking changes are made, allowing developers to update with confidenceGive us an avenue for officially deprecating and deleting old code, reducing bloatMore responsibly deal with integrations (whose SDKs are often changing as rapidly as LangChain)Even after we release a 0.2 version, we will commit to maintaining a branch of 0.1, but will only patch critical bug fixes. See more towards the end of this post on our plans for that.While re-architecting the package towards a path to a stable 0.1 release, we took the opportunity to talk to hundreds of developers about why they use LangChain and what they love about it. This input guided our direction and focus. We also used it as an opportunity to bring parity to the Python and JavaScript versions in the core areas outlined below. \uf8ffüí°While certain integrations and more tangential chains may be language specific, core abstractions and key functionality are implemented equally in both the Python and JavaScript packages.We want to share what we‚Äôve heard and our plan to continually improve LangChain. We hope that sharing these learnings will increase transparency into our thinking and decisions, allowing others to better use, understand, and contribute to LangChain. After all, a huge part of LangChain is our community ‚Äì both the user base and the 2000+ contributors ‚Äì and we want everyone to come along for the journey.\xa0Third Party IntegrationsOne of the things that people most love about LangChain is how easy we make it to get started building on any stack. We have almost 700 integrations, ranging from LLMs to vector stores to tools for agents to use. \uf8ffüí°LangChain is often used as the ‚Äúglue‚Äù to join all the different pieces you need to build an LLM app together, and so prioritizing a robust integration ecosystem is a priority for us.About a month ago, we started making some changes we think will improve the robustness, stability, scalability, and general developer experience around integrations. We split out ALL third party integrations into")]
})


# ## 3.2 Create the RAG Chain
# 
# RAG stands for Retrieval-Augmented Generation. This means that we will enrich the prompt that we send to the LLM. We will use with the documents that wil will retrieve from the vector store for this. LangChain comes with the function `create_retrieval_chain` that allows you to create one of these.

# In[23]:


# create retrieval chain

from langchain.chains import create_retrieval_chain

retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# In[24]:


response = retrieval_chain.invoke({
    "input": "What is langsmith?"
})


# In[25]:


response['answer']


# # 4.0 Create Conversational RAG Chain
# 
# Now we will create exactly the same thing as above, but we will have the AI assistant take the history of the conversation into account. In short, we will build the same chain as above but with we will take into account the previous messages in these two steps of the chain:
# 
# - When fetching the documents from the vector store. We will fetch documents related to the entire conversation and not just the latest message.
# - When answering the question. We will send to the LLM the history of the conversation along the context and query.
# 
# 
# 
# 

# ## 4.1 Create a Conversation-Aware Retrieval Chain
# 
# This chain will return the documents related to the entire conversation and not just the latest message.

# In[26]:


# conversational retrieval chain

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
    ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])

retriever_chain = create_history_aware_retriever(llm, retriever, prompt)


# In[27]:


from langchain_core.messages import HumanMessage, AIMessage

chat_history = [
    HumanMessage(content="Is there anything new about Langchain 0.1.0?"),
    AIMessage(content="Yes!")
]

retriever_chain.invoke({
    "chat_history": chat_history,
    "input": "Tell me more about it!"
})


# ## 4.2 Use Retrieval Chain together with Document Chain
# 
# Now we will create a document chain that contains a placeholder for the conversation history. This placeholder will be populated with the conversation history that we will pass as its value. We will the plug it together with the retriever chain we created above to have a conversational retrieval-augmented chain.

# In[28]:


from langchain.chains import create_retrieval_chain

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's questions based on the below context:\n\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

document_chain = create_stuff_documents_chain(llm, prompt)

conversational_retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)


# In[29]:


response = conversational_retrieval_chain.invoke({
    'chat_history': [],
    "input": "langsmith is used for debugging?"
})


# In[30]:


response


# In[31]:


response['answer']


# In[32]:


# simulate conversation history

chat_history = [
    HumanMessage(content="The main way we’ve tackled this is by building LangSmith. One of the main value props that LangSmith provides is a best-in-class debugging experience for your LLM application. We log exactly what steps are happening, what the inputs of each step are, what the outputs of each step are, how long each step takes, and more data."),
    AIMessage(content="Yes!")
]

response = conversational_retrieval_chain.invoke({
    'chat_history': chat_history,
    "input": "langsmith is used for debugging?"
})


# In[33]:


response


# In[34]:


response['answer']


# In[35]:


get_ipython().system('pip install PyPDF2')


# In[36]:


from langchain_community.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from google.colab import drive
from google.colab import auth
import os

# 1. Authenticate and mount Google Drive (mounting is not strictly necessary for GoogleDriveLoader but can be helpful for verification)
auth.authenticate_user()
drive.mount('/content/drive')

# 2. Configure Google Drive Loader
# Replace "YOUR_GOOGLE_DRIVE_FOLDER_ID" with the actual ID of your Google Drive folder.
# You can find the folder ID in the URL when you open the folder in Google Drive.
folder_id = "1jrc0R7389NTRhaetTULTuYECDm6TRnK2"
loader = GoogleDriveLoader(
    folder_id=folder_id,
    recursive=True,
    file_types=["document", "pdf", "sheet"]
)

# 3. Load and process documents
documents = loader.load()

# 4. Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
splits = text_splitter.split_documents(documents)

# 5. Create vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(splits, embeddings)

# 6. Set up conversation chain
llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo")
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key='answer'
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=memory,
    return_source_documents=True
)

# 7. Chat interface
def chat_interface():
    print("Document Q&A System (Type 'exit' to quit)")
    while True:
        query = input("\nQuestion: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break

        result = qa_chain({"question": query})

        print(f"\nAnswer: {result['answer']}")
        print("\nSources:")
        for doc in result['source_documents']:
            # Assuming 'source' metadata exists and 'page' might not always
            source_info = doc.metadata.get('source', 'Unknown source')
            page_info = doc.metadata.get('page', 'N/A')
            print(f"- {source_info} (page {page_info})")


# 8. Run the system
if __name__ == "__main__":
    # Set your OpenAI API key - Make sure to replace "your-api-key-here" with your actual key or use Colab secrets
    # It's recommended to use Colab secrets for sensitive information like API keys.
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here" # Using environment variable directly is less secure

    # Example of using Colab secrets:
    # from google.colab import userdata
    # os.environ["OPENAI_API_KEY"] = userdata.get('OPENAI_API_KEY') # Replace 'OPENAI_API_KEY' with the name of your secret

    # For this example, I'll keep the direct assignment as in the original code,
    # but strongly advise using Colab secrets in a real application.
    os.environ["OPENAI_API_KEY"] = "TOKEN"


    chat_interface()


# 
# 
