from fastapi import FastAPI
import uvicorn

app = FastAPI(title="Dessert Classification Server")


@app.get("/")
def home():
    return "Welcome to the Dessert Classification Website! You can head to http://localhost:8000/docs for testing."


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
