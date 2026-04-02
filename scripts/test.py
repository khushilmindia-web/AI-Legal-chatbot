# # # # import os
# # # # from groq import Groq

# # # # client = Groq(api_key="YOUR_API_KEY")

# # # # try:
# # # #     response = client.chat.completions.create(
# # # #         model="llama3-8b-8192",
# # # #         messages=[{"role": "user", "content": "Hello"}]
# # # #     )
# # # #     print(response.choices[0].message.content)

# # # # except Exception as e:
# # # #     print("❌ FULL ERROR:", repr(e))

# # # import os
# # # from dotenv import load_dotenv

# # # load_dotenv()

# # # print("KEY FROM ENV:", os.getenv("GROQ_API_KEY"))


# # from groq import Groq
# # import os
# # from dotenv import load_dotenv

# # load_dotenv()

# # api_key = os.getenv("GROQ_API_KEY").strip()

# # client = Groq(api_key=api_key)

# # response = client.chat.completions.create(
# #     model="mixtral-8x7b-32768",
# #     messages=[{"role": "user", "content": "Say hello formally like a lawyer"}]
# # )

# # print(response.choices[0].message.content)

# from groq import Groq
# import os
# from dotenv import load_dotenv

# load_dotenv()

# api_key = os.getenv("GROQ_API_KEY").strip()

# client = Groq(api_key=api_key)

# response = client.chat.completions.create(
#     model="llama3-8b-8192",   # ✅ FIXED MODEL
#     messages=[
#         {"role": "user", "content": "Say hello formally like a lawyer"}
#     ]
# )


# print(response.choices[0].message.content)


from groq import Groq
import os
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / "config" / ".env")

api_key = os.getenv("GROQ_API_KEY").strip()

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model="llama-3.1-8b-instant",   # ✅ UPDATED MODEL
    messages=[
        {"role": "user", "content": "Say hello formally like a lawyer"}
    ]
)

print(response.choices[0].message.content)
