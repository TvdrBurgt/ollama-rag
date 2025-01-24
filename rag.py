from openai import OpenAI

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

# Generate a response
response = client.chat.completions.create(
  model="deepseek-r1:1.5b",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)
print(response.choices[0].message.content)

# Generate embeddings
embeddings = client.embeddings.create(
  model="all-minilm",
  input="The sky is blue because of Rayleigh scattering"
)
print(embeddings.data[0].embedding)
