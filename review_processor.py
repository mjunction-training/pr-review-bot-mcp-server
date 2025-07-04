import os
import re
import logging
import requests
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Any


logger = logging.getLogger(__name__)

MAX_DIFF_LENGTH = 100000  # 100K characters
CHUNK_SIZE = 5000  # Chunk size for processing
HUGGING_FACE_API_URL = "https://api-inference.huggingface.co/models/"
HUGGING_FACE_API_TOKEN = os.getenv("HUGGING_FACE_API_TOKEN")
if not HUGGING_FACE_API_TOKEN:
    logging.error("HUGGING_FACE_API_TOKEN environment variable not set. Exiting.")
    exit(1)

class ReviewProcessor:
    def __init__(self):
        self.hugging_face_api_token = os.getenv("HUGGING_FACE_API_TOKEN")
        self.parser = StrOutputParser()

        if not self.hugging_face_api_token:
            logging.error("HUGGING_FACE_API_TOKEN environment variable not set. Exiting.")
            raise ValueError("Missing Hugging Face API token")

    def split_diff(self, diff: str) -> List[str]:
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

    def query_huggingface(self, payload: Dict, model_name: str) -> Dict:
        """Query Hugging Face Inference API"""
        headers = {"Authorization": f"Bearer {self.hugging_face_api_token}"}
        response = requests.post(
            HUGGING_FACE_API_URL + model_name,
            headers=headers,
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        return response.json()

    def parse_review_output(self, text: str) -> tuple[List[Dict], List[Dict]]:
        """Parse raw text output into structured comments and security issues."""
        comments = []
        security_issues = []
        
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith("SECURITY:"):
                parts = line.split(':', 3)
                if len(parts) >= 4:
                    try:
                        security_issues.append({
                            "file": parts[1].strip(),
                            "line": int(parts[2].strip()),
                            "issue": parts[3].strip()
                        })
                    except ValueError:
                        logger.warning(f"Could not parse security issue line: {line}")
                else:
                    logger.warning(f"Malformed security issue line: {line}")
            elif ':' in line and line.count(':') >= 2:
                parts = line.split(':', 2)
                if len(parts) == 3:
                    file_part, line_part, comment = parts
                    if file_part.strip() and line_part.strip().isdigit():
                        try:
                            comments.append({
                                "file": file_part.strip(),
                                "line": int(line_part.strip()),
                                "comment": comment.strip()
                            })
                        except ValueError:
                            logger.warning(f"Could not parse comment line: {line}")
                    else:
                        logger.warning(f"Malformed comment line (file or line missing/invalid): {line}")
                else:
                    logger.warning(f"Malformed comment line (incorrect number of colons): {line}")
        
        return comments, security_issues

    async def process_review(self, diff: str, repo: str, pr_id: int, metadata: Dict[str, Any], review_prompt_content: str, summary_prompt_content: str) -> tuple[List[Dict], str, List[Dict]]:
        """
        Process a PR review by chunking the diff, querying Hugging Face,
        and extracting structured comments and a summary.
        """
        all_comments = []
        all_security_issues = []
        
        diff_chunks = self.split_diff(diff)
        logger.info(f"Processing PR #{pr_id} from {repo}. Diff split into {len(diff_chunks)} chunks.")

        for i, chunk in enumerate(diff_chunks):
            logger.info(f"Reviewing chunk {i+1}/{len(diff_chunks)} for PR #{pr_id}.")
            
            review_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", review_prompt_content),
                    ("human", chunk)
                ]
            )
            
            review_response = self.query_huggingface(
                payload={
                    "inputs": self.parser.parse(review_prompt.format_messages(guidelines="", diff_chunk=chunk)[0].content)
                },
                model_name=os.getenv("HUGGING_FACE_REVIEW_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
            )
            
            if review_response and isinstance(review_response, list) and review_response[0] and "generated_text" in review_response[0]:
                chunk_review_text = review_response[0]["generated_text"]
                logger.debug(f"Raw review output for chunk {i+1}: {chunk_review_text}")
                
                comments, security_issues = self.parse_review_output(chunk_review_text)
                all_comments.extend(comments)
                all_security_issues.extend(security_issues)
            else:
                logger.warning(f"No valid review response for chunk {i+1} of PR #{pr_id}.")

        comments_text = "\\n".join([c["comment"] for c in all_comments])
        security_issues_text = "\\n".join([s["issue"] for s in all_security_issues])
        
        full_comments_for_summary = f"Comments:\\n{comments_text}\\nSecurity Issues:\\n{security_issues_text}"
        
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", summary_prompt_content)
            ]
        )
        
        summary_response = self.query_huggingface(
            payload={
                "inputs": self.parser.parse(summary_prompt.format_messages(comments_text=full_comments_for_summary)[0].content)
            },
            model_name=os.getenv("HUGGING_FACE_SUMMARY_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        )

        summary = "No summary generated."
        if summary_response and isinstance(summary_response, list) and summary_response[0] and "generated_text" in summary_response[0]:
            summary = summary_response[0]["generated_text"].strip()
            logger.info(f"Generated summary for PR #{pr_id}.")
        else:
            logger.warning(f"No valid summary response for PR #{pr_id}.")

        return all_comments, summary, all_security_issues
