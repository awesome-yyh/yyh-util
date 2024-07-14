from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://localhost:8000/v1"

# openai_api_key = ""

client = OpenAI(
    api_key=openai_api_key,
    # base_url=openai_api_base,
)
models = client.models.list()
print(f"models_list: {models}")
model = models.data[0].id

model = "gpt-3.5-turbo"
model = "gpt-4"

stream = False
completion = client.completions.create(
    model=model,
    prompt="A robot may not injure a human being",
    echo=False,
    n=2,
    stream=stream,
    logprobs=3
)

print("Completion results:")
if stream:
    for c in completion:
        print(c)
else:
    print(completion)

chat_completion = client.chat.completions.create(
    model=model,
    messages=[{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "Who won the world series in 2020?"
    }, {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020."
    }, {
        "role": "user",
        "content": "Where was it played?"
    }]
)

print(f"Chat completion results: {chat_completion}")
print(chat_completion.choices[0].message)
