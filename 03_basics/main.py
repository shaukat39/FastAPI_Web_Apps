from fastapi import FastAPI

app = FastAPI() # Create an instance of FastAPI

# Define a route
@app.get("/")
async def root():
    return {"message": "Hello World we are learning FastAPI"}