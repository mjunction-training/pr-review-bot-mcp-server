import os
import logging
import requests
from typing import Dict
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

logger = logging.getLogger(__name__)

class ReviewProcessor:
    def __init__(self):
        self.llm_api_base_url = os.getenv("LLM_API_BASE_URL")
        self.llm_api_token = os.getenv("LLM_API_TOKEN")
        self.llm_api_timeout = int(os.getenv("LLM_API_TIMEOUT", 600))
        self.llm_api_retries = int(os.getenv("LLM_API_RETRIES", 3))
        self.llm_api_retry_delay = int(os.getenv("LLM_API_RETRY_DELAY", 2))

        if not self.llm_api_base_url:
            logging.error("LLM_API_BASE_URL environment variable not set. Exiting.")
            raise ValueError("Missing LLM API base URL")

        if not self.llm_api_token:
            logging.error("LLM_API_TOKEN environment variable not set. Exiting.")
            raise ValueError("Missing LLM API token")

        logger.info(f"ReviewProcessor initialized with LLM API Base URL: {self.llm_api_base_url}")
        logger.debug(
            f"LLM API Timeout: {self.llm_api_timeout}s, Retries: {self.llm_api_retries}, Retry Delay: {self.llm_api_retry_delay}s")

    @retry(stop=stop_after_attempt(int(os.getenv("LLM_API_RETRIES", 3))),
           wait=wait_fixed(int(os.getenv("LLM_API_RETRY_DELAY", 2))),
           retry=retry_if_exception_type(requests.exceptions.RequestException),
           reraise=True)
    def invoke_llm_model(self, payload: Dict, model_name: str) -> Dict:
        response = {}
        headers = {"Authorization": f"Bearer {self.llm_api_token}"}

        base_url = self.llm_api_base_url.rstrip('/')
        api_url = f"{base_url}/{model_name}"

        logger.debug(
            f"Invoking LLM model at {api_url} with timeout {self.llm_api_timeout}s. Payload keys: {list(payload.keys())}")
        try:
            response = requests.post(
                api_url,
                headers=headers,
                json=payload,
                timeout=self.llm_api_timeout
            )
            response.raise_for_status()
            logger.debug(f"LLM model {model_name} invocation successful. Status: {response.status_code}")
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM API request failed for model {model_name} at {api_url}: {e}", exc_info=True)
            if response is not None:
                logger.error(f"LLM API response content: {response.text}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred while invoking LLM: {e}", exc_info=True)
            raise