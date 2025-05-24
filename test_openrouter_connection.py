import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
YOUR_SITE_URL = os.getenv("YOUR_SITE_URL", "http://localhost:8501")
YOUR_SITE_NAME = os.getenv("YOUR_SITE_NAME", "QwenQuill_Test") 

if not OPENROUTER_API_KEY:
    print("Error: OPENROUTER_API_KEY not found in .env file.")
else:
    try:
        print("Attempting to connect to OpenRouter...")

        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )

        print(f"Sending request to model: qwen/qwen3-32b")
        print(f"Using site URL: {YOUR_SITE_URL} and site name: {YOUR_SITE_NAME}")

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": YOUR_SITE_URL,
                "X-Title": YOUR_SITE_NAME,
            },
            model="qwen/qwen3-32b",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Please respond concisely."
                },
                {
                    "role": "user",
                    "content": "What is the capital of France? Respond in one sentence."
                }
            ],
            max_tokens=150,
            temperature=0.7,
        )

        if completion.choices and len(completion.choices) > 0:
            response_content = completion.choices[0].message.content
            print("\nSuccessfully received response:")
            print(f"Qwen 3 32B: {response_content}")
            print("\nFull API Response (for debugging):")
            print(completion.model_dump_json(indent=2))
        else:
            print("\nNo response choices received.")
            print("\nFull API Response (for debugging):")
            print(completion.model_dump_json(indent=2))

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        if hasattr(e, 'response') and e.response:
            try:
                print(f"Error details: {e.response.json()}")
            except:
                print(f"Error details (raw): {e.response.text}")