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
    # Load guidelines
    guidelines = load_guidelines()
    
    # Get Hugging Face API token
    hf_token = os.getenv("HUGGING_FACE_API_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_API_TOKEN not set")
        raise ValueError("Missing Hugging Face API token")
    
    # Choose a free model (small and efficient)
    # model_name = "codellama/CodeLlama-7b-instruct-hf"  # 7B parameter model
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # <--- Change this line
    
    # Split large diffs
    diff_chunks = split_diff(diff)
    all_comments = []
    security_issues = []
    
    # Process each chunk
    for i, chunk in enumerate(diff_chunks):
        # Construct prompt
        prompt = f"""<s>[INST] <<SYS>>
You are an expert code reviewer. Follow these guidelines:
{guidelines}

Review tasks:
1. Summarize changes in this diff chunk
2. Add line comments (format: FILE:LINE: COMMENT)
3. Flag security vulnerabilities (format: SECURITY:FILE:LINE: ISSUE)
<</SYS>>

Review this code diff:

{chunk} [/INST]"""
        
        # Query Hugging Face API
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1000,
                "temperature": 0.2,
                "top_p": 0.95,
                "return_full_text": False
            }
        }
        
        try:
            response = query_huggingface(payload, model_name, hf_token)
            generated_text = response[0]['generated_text']
        except Exception as e:
            logger.error(f"Model query failed: {str(e)}")
            continue
        
        # Parse response
        comments, security = parse_response(generated_text)
        all_comments.extend(comments)
        security_issues.extend(security)
    
    # Generate summary
    summary_payload = {
        "inputs": f"Generate concise summary of PR #{pr_id} in {repo} based on these comments:\n\n" +
                  "\n".join([c['comment'] for c in all_comments]),
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.1
        }
    }
    
    try:
        summary_response = query_huggingface(summary_payload, model_name, hf_token)
        summary = summary_response[0]['generated_text']
    except Exception as e:
        logger.error(f"Summary generation failed: {str(e)}")
        summary = "PR summary could not be generated"
    
    return {
        "summary": summary,
        "comments": all_comments,
        "security_issues": security_issues
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