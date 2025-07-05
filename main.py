import logging
import os
from typing import List, Dict, Any

import requests
from fastmcp import FastMCP
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.responses import Response

from review_processor import ReviewProcessor

logging.basicConfig(level=logging.DEBUG)

class ReviewInput(BaseModel):
    """Schema for the input payload to the PR review model."""
    diff: str
    repo: str
    pr_id: int
    metadata: Dict[str, Any]
    review_prompt_content: str
    summary_prompt_content: str

class Comment(BaseModel):
    """Schema for a single line comment."""
    file: str
    line: int
    comment: str

class SecurityIssue(BaseModel):
    """Schema for a single security issue."""
    file: str
    line: int
    issue: str

class ReviewOutput(BaseModel):
    """Schema for the output payload from the PR review model."""
    summary: str
    comments: List[Comment]
    security_issues: List[SecurityIssue]

# Initialize FastMCP server
mcp = FastMCP(name="PR Review MCP Server", host="0.0.0.0")

review_processor_instance = ReviewProcessor()

# Define the handler function as a FastMCP tool
@mcp.tool(name="pr_review_model")
async def pr_review_handler(input_data: ReviewInput) -> ReviewOutput:
    """
    Handler function for the PR review model exposed via FastMCP.
    It takes a ReviewInput object and returns a ReviewOutput object.
    """
    try:
        comments_data, summary, security_issues_data = await review_processor_instance.process_review(
            diff=input_data.diff,
            repo=input_data.repo,
            pr_id=input_data.pr_id,
            metadata=input_data.metadata,
            review_prompt_content=input_data.review_prompt_content,
            summary_prompt_content=input_data.summary_prompt_content
        )
        
        comments = [Comment(**c) for c in comments_data]
        security_issues = [SecurityIssue(**s) for s in security_issues_data]

        response_model = ReviewOutput( #
            summary=summary,
            comments=comments,
            security_issues=security_issues
        )

        return response_model.model_dump()
    except requests.exceptions.RequestException as e:
        logging.error(f"Hugging Face API request failed: {e}")
        # FastMCP tools handle exceptions. Re-raising a generic RuntimeError for the tool.
        raise RuntimeError(f"External service error: {e}")
    except Exception as e:
        logging.error(f"Error processing review request: {e}")
        raise RuntimeError(f"Internal server error: {e}")

# Define the health check endpoint using FastMCP's custom_route
@mcp.custom_route("/health", methods=["GET"])
async def health_check_mcp() -> Response:
    """
    Health check endpoint for the MCP server.
    """
    status = {
        "status": "ok",
        "services": {}
    }

    hf_token = os.getenv("HUGGING_FACE_API_TOKEN")
    try:
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
    except requests.exceptions.RequestException as e:
        logging.error(f"Hugging Face API check failed: {e}")
        status["services"]["hugging_face"] = "error"
    except Exception as e:
        logging.error(f"Unexpected error during Hugging Face API health check: {e}")
        status["services"]["hugging_face"] = "error"
    
    # Return a PlainTextResponse with JSON content for custom routes
    return JSONResponse(status, media_type="application/json")


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    # Use mcp.run() to start the FastMCP server
    mcp.run(transport="http", port=port, path="/mcp")
