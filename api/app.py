from fastapi import FastAPI

from api.routes import router


app = FastAPI(
    title="Research Agent Chatbot",
)

app.include_router(
    router,
)