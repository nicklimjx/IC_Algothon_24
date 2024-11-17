from fastapi import FastAPI, HTTPException
import json

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/latest-weights", summary="Retrieve the latest submission weights")
def get_latest_weights():
    """
    Endpoint to retrieve the latest submission weights.
    Calls the get_latest() function to fetch data from the CSV.
    """
    data = get_latest()
    
    # Check for errors in the returned data
    if "error" in data:
        raise HTTPException(status_code=500, detail=data["error"])
    
    return data

def get_latest():
    with open('submission.json', 'r') as f:
        data = json.load(f)
    return data