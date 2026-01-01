import json
import logging
import os
import asyncio
import atexit
from typing import Any, Optional, Tuple
from uuid import uuid4
import time

from tools.base_tool import BaseTool
from schemas.tool_schemas import OpenAIFunctionToolSchema
from core.rollout_trace import rollout_trace_op

import httpx

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class ESSearchToolNoSummary(BaseTool):
    """Tool for performing searches without summary (for stress testing)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.search_es_url = os.getenv("SEARCH_SERVICE_URL", os.getenv("SEARCH_ES_URL", "http://localhost:5003/serper"))
        max_connections = 10000
        limits = httpx.Limits(
            max_keepalive_connections=max_connections, 
            max_connections=max_connections, 
            keepalive_expiry=60
        )
        timeout = httpx.Timeout(
            connect=300.0,
            read=300.0,
            write=30.0,
            pool=10.0
        )
        self._httpx_client = httpx.AsyncClient(limits=limits, verify=False, timeout=timeout)
        atexit.register(self.on_exit)

    def on_exit(self):
        """Clean up httpx client during program shutdown."""
        try:
            if hasattr(self, "_httpx_client") and self._httpx_client is not None:
                try:
                    import threading
                    def background_cleanup():
                        try:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            loop.run_until_complete(self._httpx_client.aclose())
                            loop.close()
                        except Exception:
                            pass
                        finally:
                            self._httpx_client = None
                    cleanup_thread = threading.Thread(target=background_cleanup, daemon=True)
                    cleanup_thread.start()
                except Exception:
                    self._httpx_client = None
        except Exception:
            pass

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
        """Execute search without summary."""
        try:
            query_input = parameters.get("query", "")
        except Exception as e:
            logger.error(f"[Search params error]: {e}")
            logger.error(f"parameters: {parameters}")
            return "[Search] Invalid request format: Input must be a JSON object containing 'query' field", 0.0, {}

        self._instance_dict[instance_id]["operation_count"] += 1

        search_results = await self._google_search(query_input)

        self._instance_dict[instance_id]["current_result"] = search_results.strip()

        reward = 0.0

        metrics = {
            "query": query_input,
            "response": search_results,
            "operation_count": self._instance_dict[instance_id]["operation_count"],
        }

        return search_results.strip(), reward, metrics

    async def _google_search(self, query, k: int = 10) -> str:
        """Perform search using ES API without summary."""
        async def single_search(q: str) -> str:
            payload = {"query": q}
            print(f"[🔍 Search-ES] Query: {q}")
            t0 = time.perf_counter()
            print(f"[🔍 Search-ES] Calling service: {self.search_es_url}")
            
            resp = await self._httpx_client.post(self.search_es_url, json=payload)
            
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[🔍 Search-ES] Service call completed in {elapsed:.3f}s")
            logger.info(f"Search API call took {elapsed:.3f}s")
            data = resp.json()
            if resp.status_code != 200:
                raise RuntimeError(f"Search API error {resp.status_code}: {data}")

            organic = data.get("organic", [])
            if not organic:
                print(f"[⚠️ Search-ES] No results found for query: '{q}'")
                raise Exception(f"No results found for query: '{q}'. Use a less specific query.")

            snippets = [
                f"{idx}. [{item.get('title', 'No title')}]({item.get('link', '')})\n{item.get('snippet', '')}"
                for idx, item in enumerate(organic[:k], 1)
            ]
            print(f"[✅ Search-ES] Found {len(snippets)} results for query: '{q}'")
            return (
                f"A Google search for '{q}' found {len(snippets)} results:\n\n"
                "## Web Results\n" + "\n\n".join(snippets)
            )

        # Handle query parsing
        queries = [query] if isinstance(query, str) else query
        
        if isinstance(queries, list) and len(queries) == 1 and isinstance(queries[0], str):
            try:
                parsed_content = json.loads(queries[0])
                if isinstance(parsed_content, list):
                    queries = parsed_content
                    logger.info(f"[Search] Successfully parsed JSON list: {queries}")
            except (json.JSONDecodeError, TypeError):
                if queries[0].startswith('[') and queries[0].endswith(']'):
                    cleaned_content = queries[0][1:-1]
                    queries = [cleaned_content]
                    logger.info(f"[Search] Stripped brackets: {queries}")
                else:
                    logger.warning(f"[Search] Failed to parse query, keeping original: {queries}")

        print(f"[🔍 Search-ES] Processing {len(queries)} query/queries")
        logger.debug(f"[Search] Final queries list: {queries}")
        tasks = [asyncio.create_task(single_search(q)) for q in queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to strings
        final = []
        for q, res in zip(queries, results):
            if isinstance(res, Exception):
                error_msg = f"[Search] Failed to Search for '{q}': {res}"
                print(f"[❌ Search-ES] {error_msg}")
                final.append(error_msg)
                logger.error(f"[Search] Failed to Search for '{q}': {res}")
            else:
                final.append(res)
        
        result_text = "\n=======\n".join(final)
        print(f"[✅ Search-ES] Completed, returned {len(final)} result(s)")
        return result_text

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate tool reward."""
        return 0.0

