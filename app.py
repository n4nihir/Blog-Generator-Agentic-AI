import uvicorn
from fastapi import FastAPI, Request
from src.graphs.graph_builder import GraphBuilder
from src.llms.openaillm import OpenAILLM

import os
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## APIs

@app.post("/blogs")
async def generate_blog(request: Request):
    data = await request.json()
    topic = data.get("topic")
    language = data.get("current_language")

    if not topic:
        return JSONResponse(content={"error": "Topic is required"}, status_code=400)
    
    openai_llm = OpenAILLM().get_llm()

    graph_builder = GraphBuilder(openai_llm)

    if topic and not language:
        graph = graph_builder.setup_graph(usecase="topic")
        state = graph.invoke({"topic": topic})
    elif topic and language:
        graph = graph_builder.setup_graph(usecase="language")
        state = graph.invoke({"topic": topic, "current_language": language.lower()})

    return {"data": state}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)