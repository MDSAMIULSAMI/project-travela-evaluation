from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

response = client.moderations.create(
    model="omni-moderation-latest",
    input="Son of a bitch",
)

print(response)