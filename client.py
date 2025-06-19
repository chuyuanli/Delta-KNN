import os
from openai import OpenAI


def send_chat_request(openai_api_key, openai_api_base, model, messages, max_new_tokens, temperature, top_p, seed):
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=False,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed
    )
    return chat_completion.choices[0].message.content

def send_embedding_request(text):
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding