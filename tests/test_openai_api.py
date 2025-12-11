from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
client = OpenAI()
resp = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role":"user","content":"Ping"}],
    max_tokens=10,
)
print(resp.choices[0].message.content)