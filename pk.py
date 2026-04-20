import os
from portkey_ai import Portkey

portkey = Portkey(
  api_key = os.environ["PORTKEY_API_KEY"]
)

response = portkey.chat.completions.create(
    model = "@deepinfra/Qwen/Qwen2.5-7B-Instruct",
    messages = [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Portkey"}
    ],
    MAX_TOKENS = 512
)

print(response.choices[0].message.content)