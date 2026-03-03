from fastapi import FastAPI
from api.routers import inventory

app = FastAPI(title="Elite Supply Chain Platform")

# This line connects the file you just made in api/routers/
app.include_router(inventory.router)

@app.get("/")
def root():
    return {"message": "Supply Chain API is Live", "docs": "/docs"}