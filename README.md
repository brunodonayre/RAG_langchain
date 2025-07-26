
# Sistema RAG con LangChain

Este repositorio contiene un ejemplo práctico de un sistema **RAG (Retrieval-Augmented Generation)** utilizando la librería [LangChain](https://www.langchain.com/) para conectar modelos de lenguaje con fuentes externas de conocimiento.

## 📁 Archivos

- `13_1_Langchain_RAG.py`: Script principal con la implementación de RAG usando LangChain.
- `13_1_Langchain_RAG.ipynb`: Notebook original con explicaciones paso a paso.

## 📦 Requisitos

Instala los paquetes necesarios:

```bash
pip install langchain openai faiss-cpu tiktoken
```

También puedes usar otras librerías como:

- `chromadb` o `weaviate` si deseas otros vectores.
- `dotenv` para manejar el API key.

## ⚙️ Flujo de trabajo

1. **Carga de documentos:** Se leen archivos o textos desde una fuente local.
2. **Segmentación y Embedding:** Se convierten los textos en vectores usando un modelo de embeddings (por ejemplo `OpenAIEmbeddings`).
3. **Almacenamiento:** Los vectores se almacenan en una base de datos vectorial como FAISS.
4. **Recuperación + Generación (RAG):** Se recuperan los textos más relevantes y se los pasa al modelo LLM para generar respuestas contextualizadas.

## 🚀 Ejemplo de uso

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Configura modelo y sistema RAG
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    retriever=vectorstore.as_retriever()
)

qa.run("¿Cuál es el propósito del documento?")
```

## 📌 Notas

- Se puede usar cualquier modelo de embeddings compatible (OpenAI, Hugging Face, Cohere, etc).
- Requiere una API Key válida de OpenAI (`OPENAI_API_KEY` en entorno o .env).

## 🧠 Sobre RAG

RAG combina recuperación de documentos relevantes con generación de texto para construir sistemas conversacionales más precisos y fundamentados.

## 📜 Licencia

MIT. Los modelos de lenguaje y bases vectoriales están sujetos a sus propias licencias.
