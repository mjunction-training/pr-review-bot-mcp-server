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

class ReviewProcessor:
    def __init__(self):
        self.hugging_face_api_token = os.getenv("HUGGING_FACE_API_TOKEN")
        self.parser = StrOutputParser()

        if not self.hugging_face_api_token:
            logging.error("HUGGING_FACE_API_TOKEN environment variable not set. Exiting.")
            raise ValueError("Missing Hugging Face API token")

    @staticmethod
    def split_diff(diff: str) -> List[str]:
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
            timeout=500
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def parse_review_output(text: str) -> tuple[List[Dict], List[Dict]]:
        """Parse raw text output into structured comments and security issues."""
        comments = []
        security_issues = []
        
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            # Updated regex to correctly capture file, line, and the full comment/issue text
            security_match = re.match(r"SECURITY:([^:]+):(\d+):(.+)", line)
            comment_match = re.match(r"([^:]+):(\d+):(.+)", line)

            if security_match:
                try:
                    file, line_num, issue = security_match.groups()
                    security_issues.append({
                        "file": file.strip(),
                        "line": int(line_num.strip()),
                        "issue": issue.strip()
                    })
                except ValueError:
                    logger.warning(f"Could not parse security issue line: {line}")
            elif comment_match:
                try:
                    file, line_num, comment = comment_match.groups()
                    comments.append({
                        "file": file.strip(),
                        "line": int(line_num.strip()),
                        "comment": comment.strip()
                    })
                except ValueError:
                    logger.warning(f"Could not parse comment line: {line}")
            else:
                logger.warning(f"Line did not match expected comment or security issue format: {line}")
        
        return comments, security_issues

    async def process_review_no_chunk(self, diff: str, repo: str, pr_id: int, metadata: Dict[str, Any],
                             review_prompt_content: str, summary_prompt_content: str) -> tuple[
        List[Dict], str, List[Dict]]:
        """
        Process a PR review by sending the entire diff, querying Hugging Face,
        and extracting structured comments and a summary.
        """
        # Remove chunking and send the entire diff in one shot
        review_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", review_prompt_content),
                ("human", diff)  # Send the full diff directly
            ]
        )

        prompt_content_for_hf = self.parser.parse(review_prompt.format_messages()[0].content)

        review_response = self.query_huggingface(
            payload={
                "inputs": prompt_content_for_hf
            },
            model_name=os.getenv("HUGGING_FACE_REVIEW_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        )

        all_comments = []
        all_security_issues = []

        if review_response and isinstance(review_response, list) and review_response[0] and "generated_text" in \
                review_response[0]:
            review_text = review_response[0]["generated_text"]
            comments, security_issues = self.parse_review_output(review_text)
            all_comments.extend(comments)
            all_security_issues.extend(security_issues)
        else:
            logger.warning(f"No valid review response for PR #{pr_id}. Response: {review_response}")

        comments_text = "\n".join([c["comment"] for c in all_comments])
        security_issues_text = "\n".join([s["issue"] for s in all_security_issues])

        final_summary_prompt_text = f"""
               {summary_prompt_content}

               <comments_and_security_issues>
               Comments:
               {comments_text}
               Security Issues:
               {security_issues_text}
               </comments_and_security_issues>
               """

        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", final_summary_prompt_text)
            ]
        )

        summary_response = self.query_huggingface(
            payload={
                "inputs": self.parser.parse(summary_prompt.format_messages()[0].content)
            },
            model_name=os.getenv("HUGGING_FACE_SUMMARY_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
        )

        summary = "No summary generated."
        if summary_response and isinstance(summary_response, list) and summary_response[0] and "generated_text" in \
                summary_response[0]:
            summary = summary_response[0]["generated_text"].strip()
        else:
            logger.warning(f"No valid summary response for PR #{pr_id}. Response: {summary_response}")

        return all_comments, summary, all_security_issues


    async def process_review(self, diff: str, repo: str, pr_id: int, metadata: Dict[str, Any], review_prompt_content: str, summary_prompt_content: str) -> tuple[List[Dict], str, List[Dict]]:
        """
        Process a PR review by chunking the diff, querying Hugging Face,
        and extracting structured comments and a summary.
        """
        all_comments = []
        all_security_issues = []

        diff_chunks = self.split_diff(diff)
        logger.info(f"Processing PR #{pr_id} from {repo}. Diff split into {len(diff_chunks)} chunks.")

        # Note: The review_prompt_content from mcp_client.py now contains the full system message.
        # The human message for this prompt is the 'chunk' itself.
        # No need to format review_prompt_content here with guidelines or diff_chunk as it's already a full string.
        
        for i, chunk in enumerate(diff_chunks):
            logger.info(f"Reviewing chunk {i+1}/{len(diff_chunks)} for PR #{pr_id}.")
            #logger.info(f"Querying Hugging Face Inference API. {review_prompt_content}, {chunk}")
            review_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", review_prompt_content),
                    ("human", chunk)
                ]
            )
            
            review_response = self.query_huggingface(
                payload={
                    "inputs": self.parser.parse(review_prompt.format_messages()[0].content) # No extra args needed as review_prompt is fully formed
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

        comments_text = "\n".join([c["comment"] for c in all_comments])
        security_issues_text = "\n".join([s["issue"] for s in all_security_issues])
        
        # IMPORTANT CHANGE: Construct the full summary prompt text before creating ChatPromptTemplate
        final_summary_prompt_text = f"""
            {summary_prompt_content}

            <comments_and_security_issues>
            Comments:
            {comments_text}
            Security Issues:
            {security_issues_text}
            </comments_and_security_issues>
            """
        
        summary_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", final_summary_prompt_text) # Pass the fully formatted string
            ]
        )
        
        summary_response = self.query_huggingface(
            payload={
                "inputs": self.parser.parse(summary_prompt.format_messages()[0].content) # No args needed, prompt is full
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
