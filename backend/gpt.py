from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
EMBEDDING_MODEL = "text-embedding-3-large"

def answer_question(question, context_chunks, llm_model="gpt-3.5-turbo"):
    context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
    prompt = f"""
    You are an AI assistant. Use the following context from documents to answer the question. 
    If the answer is not in the context, say "I don't know".

    Context:
    {context_text}

    Question:
    {question}

    Answer:
    """
    response = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

def llm_generate(prompt: str, llm_model: str = "gpt-4o-mini") -> str:
    try:
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error in llm_generate: {e}")
        return ""

def get_embedding(text: str) -> list:
    try:
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error in get_embedding: {e}")
        return []
