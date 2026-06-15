from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFaceEndpoint (
    repo_id="meta-llama/Llama-3.1-8B-Instruct",
    task="text-generation"
)
# print(llm)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is this model which i am talking to?")

print(result.content)