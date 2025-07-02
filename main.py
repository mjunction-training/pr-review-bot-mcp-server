import os
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

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)