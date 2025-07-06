import logging
import os
from typing import Dict, Any

import requests
from fastmcp import FastMCP
from pydantic import BaseModel
from starlette.responses import JSONResponse, Response

from review_processor import ReviewProcessor

if not logging.root.handlers:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_log_level = getattr(logging, log_level, None)
    if not isinstance(numeric_log_level, int):
        numeric_log_level = logging.INFO
        logging.warning(f"Invalid LOG_LEVEL '{log_level}' provided. Defaulting to INFO.")
    logging.basicConfig(level=numeric_log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info(f"FastMCP server logging configured with level: {logging.getLevelName(numeric_log_level)}")


class LLMInvokeInput(BaseModel):
    model_name: str
    inputs: str

class LLMInvokeOutput(BaseModel):
    response_data: Dict[str, Any]

port = int(os.getenv("PORT", 8080))
mcp = FastMCP(name="PR Review MCP Server", host="0.0.0.0", port=port)

review_processor_instance = ReviewProcessor()
logging.info("ReviewProcessor instance initialized in main.py.")


@mcp.tool(name="llm_invoke_model")
async def llm_invoke_handler(input_data: LLMInvokeInput) -> LLMInvokeOutput:
    logging.info(f"Received LLM invocation request for model: {input_data.model_name}")
    logging.debug(f"LLM invocation payload inputs (first 200 chars): {input_data.inputs[:200]}...")
    try:
        raw_llm_response = review_processor_instance.invoke_llm_model(
            payload={"inputs": input_data.inputs},
            model_name=input_data.model_name
        )
        logging.info(f"Successfully invoked LLM model: {input_data.model_name}")
        logging.debug(f"Raw LLM response: {raw_llm_response}")
        return LLMInvokeOutput(response_data=raw_llm_response)
    except Exception as e:
        logging.error(f"Error during LLM invocation in handler for model {input_data.model_name}: {e}", exc_info=True)
        raise RuntimeError(f"Internal server error: {e}")


@mcp.custom_route("/health", methods=["GET"])
async def health_check_mcp() -> Response:
    logging.info("Received MCP health check request.")
    status = {
        "status": "ok",
        "services": {}
    }

    llm_api_token = os.getenv("LLM_API_TOKEN")
    llm_api_base_url = os.getenv("LLM_API_BASE_URL")

    llm_api_health_check_timeout = int(os.getenv("LLM_API_HEALTH_CHECK_TIMEOUT", 3))

    try:
        if llm_api_token and llm_api_base_url:
            headers = {"Authorization": f"Bearer {llm_api_token}"}
            test_url = f"{llm_api_base_url.rstrip('/')}/"
            logging.debug(
                f"Attempting LLM API connectivity check to: {test_url} with timeout {llm_api_health_check_timeout}s")
            response = requests.get(
                test_url,
                headers=headers,
                timeout=llm_api_health_check_timeout
            )
            status["services"][
                "llm_api"] = "reachable" if response.ok else f"unreachable (status: {response.status_code})"
            logging.info(f"LLM API connectivity check status: {status['services']['llm_api']}")
            logging.debug(f"LLM API health check raw response status: {response.status_code}, text: {response.text}")
        else:
            status["services"]["llm_api"] = "not_configured"
            logging.warning("LLM_API_TOKEN or LLM_API_BASE_URL not configured for health check.")
    except requests.exceptions.RequestException as e:
        logging.error(f"LLM API health check failed: {e}", exc_info=True)
        status["services"]["llm_api"] = f"unreachable (error: {e})"
    except Exception as e:
        logging.error(f"Unexpected error during LLM API health check: {e}", exc_info=True)
        status["services"]["llm_api"] = f"unreachable (unexpected error: {e})"

    logging.info(f"Returning MCP health check status: {status['status']}")
    logging.debug(f"Full MCP health check response: {status}")
    return JSONResponse(status, media_type="application/json")


if __name__ == "__main__":
    logging.info(f"Starting FastMCP server on host 0.0.0.0, port {port}")
    mcp.run()