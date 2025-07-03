import os
import re
import logging
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import HuggingFaceEndpoint
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

MAX_DIFF_LENGTH = 100000  # 100K characters
CHUNK_SIZE = 5000  # Chunk size for processing
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/"

def load_guidelines():
    """Load guidelines from markdown file"""
    try:
        with open("guidelines.md", "r") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load guidelines: {str(e)}")
        return ""

def split_diff(diff):
    """Split large diffs into manageable chunks"""
    if len(diff) <= MAX_DIFF_LENGTH:
        return [diff]
    
    chunks = []
    current_chunk = ""
    for line in diff.split('\n'):
        if len(current_chunk) + len(line) + 1 > CHUNK_SIZE:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += line + '\n'
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

def query_huggingface(payload, model_name, api_token):
    """Query Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {api_token}"}
    response = requests.post(
        HUGGING_FACE_API_URL + model_name,
        headers=headers,
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()

def process_review(diff, repo, pr_id, metadata):
    """
    Mocks the PR review process to return a generic comment.
    """
    logger.info(f"MOCKING: Processing review for PR #{pr_id} in {repo}")
    
    # Return a generic, mocked response
    return {
        "summary": "This PR IS Reviewed By - PR BOT - Response Came From MCP Server!!",
        "comments": [], # No specific line comments in mock
        "security_issues": [] # No security issues in mock
    }

def parse_response(response):
    comments = []
    security_issues = []
    
    for line in response.split('\n'):
        if line.startswith("SECURITY:"):
            parts = line.split(':', 3)
            if len(parts) >= 4:
                security_issues.append({
                    "file": parts[1].strip(),
                    "line": int(parts[2]),
                    "issue": parts[3].strip()
                })
        elif ':' in line and line.count(':') >= 2:
            file_part, line_part, comment = line.split(':', 2)
            if file_part.strip() and line_part.strip().isdigit():
                comments.append({
                    "file": file_part.strip(),
                    "line": int(line_part.strip()),
                    "comment": comment.strip()
                })
    
    return comments, security_issues