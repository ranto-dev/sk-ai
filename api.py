from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from chatbot import chatbot_response
import uvicorn
from pydantic import BaseModel

app = FastAPI()

class Msg(BaseModel):
    msg: str

app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_credentials=True, allow_methods=['*'], allow_headers=['*'])


@app.post("/chat")
async def Chat(msg: Msg):
    response = chatbot_response(msg.msg)
    return {"response": response}

if __name__ == '__main__':
    uvicorn.run("api:app", port=8000, host="0.0.0.0", reload=True)
