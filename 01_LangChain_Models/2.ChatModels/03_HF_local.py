from langchain_huggingface import ChatHuggingFace , HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id (
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation" ,
    pipeline_kwargs= dict({"max_new_tokens": 100 , "temperature": 0.5})
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("Who is this model which i am talking to and where it is running on my machine")

print(result.content)