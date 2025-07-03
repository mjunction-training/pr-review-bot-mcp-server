import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from review_processor import process_review

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.post("/review")
async def review_pr(payload: dict):
    try:
        result = process_review(
            diff=payload['diff'],
            repo=payload['repo'],
            pr_id=payload['pr_id'],
            metadata=payload.get('metadata', {})
        )
        return {
            "summary": result["summary"],
            "comments": result["comments"],
            "security_issues": result["security_issues"]
        }
    except Exception as e:
        logging.error(f"Review failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

app = FastAPI()
logging.basicConfig(level=logging.INFO)

@app.get("/health")
def health_check():
    """Comprehensive health check"""
    status = {
        "status": "ok",
        "services": {}
    }
    
    # Check Hugging Face API
    try:
        hf_token = os.getenv("HUGGING_FACE_API_TOKEN")
        if hf_token:
            headers = {"Authorization": f"Bearer {hf_token}"}
            response = requests.get(
                "https://huggingface.co/api/whoami-v2",
                headers=headers,
                timeout=3
            )
            status["services"]["hugging_face"] = "reachable" if response.ok else "unreachable"
        else:
            status["services"]["hugging_face"] = "no_token"
    except:
        status["services"]["hugging_face"] = "error"
    
    # Check guidelines file
    try:
        with open("guidelines.md.txt", "r") as f:
            content = f.read(100)
            status["services"]["guidelines"] = "available" if content else "empty"
    except:
        status["services"]["guidelines"] = "missing"
    
    return status

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)