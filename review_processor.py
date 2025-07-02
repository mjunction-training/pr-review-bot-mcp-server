import os
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from vector_store import get_vector_store
import logging

logger = logging.getLogger(__name__)

MAX_DIFF_LENGTH = 100000  # 100K characters
CHUNK_SIZE = 4000  # For splitting large diffs

def split_diff(diff):
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

def get_relevant_guidelines(diff_chunk, vector_store):
    if not vector_store:
        return ""
    
    try:
        docs = vector_store.similarity_search(diff_chunk, k=3)
        return "\n".join([d.page_content for d in docs])
    except Exception as e:
        logger.error(f"Vector search failed: {str(e)}")
        return ""

def process_review(diff, repo, pr_id, metadata):
    # Initialize components
    llm = ChatAnthropic(
        model="claude-3-sonnet-20240229",
        temperature=0.1,
        max_tokens=4000,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
    )
    vector_store = get_vector_store()
    
    # Split large diffs
    diff_chunks = split_diff(diff)
    all_comments = []
    security_issues = []
    
    # Process each chunk
    for i, chunk in enumerate(diff_chunks):
        guidelines = get_relevant_guidelines(chunk, vector_store)
        
        # Construct prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert code reviewer. Follow these guidelines:
            {guidelines}
            
            Review tasks:
            1. Summarize changes in this diff chunk
            2. Add line comments (format: FILE:LINE: COMMENT)
            3. Flag security vulnerabilities (format: SECURITY:FILE:LINE: ISSUE)
            """),
            ("human", "Review this code diff:\n\n{diff_chunk}")
        ])
        
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({
            "guidelines": guidelines,
            "diff_chunk": chunk
        })
        
        # Parse response
        comments, security = parse_response(response)
        all_comments.extend(comments)
        security_issues.extend(security)
    
    # Generate summary
    summary_prompt = ChatPromptTemplate.from_template(
        "Generate concise summary of PR #{pr_id} in {repo}:\n\n{all_comments}"
    )
    summary_chain = summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({
        "pr_id": pr_id,
        "repo": repo,
        "all_comments": "\n".join([c['comment'] for c in all_comments])
    })
    
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