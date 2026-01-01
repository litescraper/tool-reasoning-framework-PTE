import json
import logging
import os
import asyncio
from typing import Any, Optional, Tuple
from uuid import uuid4

from tools.base_tool import BaseTool
from schemas.tool_schemas import OpenAIFunctionToolSchema
from core.rollout_trace import rollout_trace_op

import httpx
import time

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class GoogleSearchTool(BaseTool):
    """Tool for performing Google searches."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)

    async def create(
        self, instance_id: Optional[str] = None, identity: dict = None, **kwargs
    ) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "history": [],
            "current_result": "",
            "operation_count": 0,
            "ground_truth": "",
        }
        return instance_id

    @rollout_trace_op
    async def execute(
        self, instance_id: str, parameters: dict[str, Any], **kwargs
    ) -> Tuple[str, float, dict]:
        """Execute Google search."""
        try:
            query_input = parameters.get("query", "")
        except Exception as e:
            logger.error(f"[Search params error]: {e}")
            logger.error(f"parameters: {parameters}")
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", 0.0, {}

        # Update instance state
        self._instance_dict[instance_id]["operation_count"] += 1

        search_results = await self._google_search(query_input)

        # Update instance state
        self._instance_dict[instance_id]["current_result"] = search_results.strip()

        reward = 0.0

        metrics = {
            "query": query_input,
            "response": search_results,
            "operation_count": self._instance_dict[instance_id]["operation_count"],
        }

        return search_results.strip(), reward, metrics

    async def _google_search(self, query, k: int = 10) -> str:
        """Perform Google search using Serper API."""
        SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")
        url = "https://google.serper.dev/search"

        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }

        async def single_search(q: str) -> str:
            q_filtered = f"{q} -site:huggingface.co/datasets -site:huggingface.co/spaces"
            payload = {
                "q": q_filtered,
                "num": 10,
                "extendParams": {
                    "country": "en",
                    "page": 1,
                },
            }
            
            serper_service_url = os.getenv("SERPER_SERVICE_URL", "http://localhost:5003/serper")
            with httpx.Client(timeout=10) as cli:
                t0 = time.perf_counter()
                print(f"[🔍 Search] Calling search service: {serper_service_url}")
                resp = cli.post(serper_service_url, json=payload, headers=headers)
                t1 = time.perf_counter()
                elapsed = t1 - t0
                print(f"[🔍 Search] Service call completed in {elapsed:.3f}s")
                logger.info(f"Serper API call took {elapsed:.3f}s")
                
                data = resp.json()
                if resp.status_code != 200:
                    raise RuntimeError(f"Serper API error {resp.status_code}: {data}")

            organic = data.get("organic", [])
            if not organic:
                raise Exception(f"No results found for query: '{query}'. Use a less specific query.")

            snippets = [
                f"{idx}. [{item.get('title', 'No title')}]({item.get('link', '')})\n{item.get('snippet', '')}"
                for idx, item in enumerate(organic[:k], 1)
            ]
            return (
                f"A Google search for '{q}' found {len(snippets)} results:\n\n"
                "## Web Results\n" + "\n\n".join(snippets)
            )

        # Handle both single query and list of queries
        queries = [query] if isinstance(query, str) else query
        print(f"[🔍 Search] Processing {len(queries)} query/queries")
        tasks = [asyncio.create_task(single_search(q)) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to strings
        final = []
        for q, res in zip(queries, results):
            if isinstance(res, Exception):
                error_msg = f"[Search] Failed to Search for '{q}': {res}"
                print(f"[❌ Search] {error_msg}")
                final.append(error_msg)
                logger.error(f"[Search] Failed to Search for '{q}': {res}")
            else:
                final.append(res)
        
        result_text = "\n=======\n".join(final)
        print(f"[✅ Search] Completed, returned {len(final)} result(s)")
        return result_text

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate tool reward."""
        return 0.0

