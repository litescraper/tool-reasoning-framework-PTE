import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import time

from tools.base_tool import BaseTool
from schemas.tool_schemas import OpenAIFunctionToolSchema
from core.rollout_trace import rollout_trace_op

import requests

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VisitToolNoSummary(BaseTool):
    """Tool for visiting web pages without summary extraction (for stress testing)."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.jina_service_url = os.getenv("JINA_SERVICE_URL", os.getenv("VISIT_ES_URL", "http://localhost:5003/jina"))

    def _is_huggingface_dataset_or_space_url(self, url):
        """Check if the URL is a Hugging Face dataset or space URL."""
        if not url:
            return False
        return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url

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
        """Execute visit tool without summary extraction."""
        try:
            url = parameters.get("url", "")
            goal = parameters.get("goal", "")
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields", 0.0, {}
        
        if self._is_huggingface_dataset_or_space_url(url):
            logger.warning(f"[Visit] Blocked HuggingFace URL, url={url}")
            return "[Visit] You are trying to visit a Hugging Face dataset for answers, please do not use the visit tool for this purpose.", 0.0, {}
        
        self._instance_dict[instance_id]["operation_count"] += 1
        print(f"[🌐 Visit-ES] Starting visit to: {url}")
        print(f"[🌐 Visit-ES] Goal: {goal}")
        
        try:
            t0 = time.perf_counter()
            print(f"[🌐 Visit-ES] Calling Jina service: {self.jina_service_url}")
            r1 = requests.post(self.jina_service_url, json={'url': url}, timeout=30)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[🌐 Visit-ES] Service call completed in {elapsed:.3f}s")
            logger.info(f"[Visit] Success, took {elapsed:.3f}s, url={url}")
            
            r1 = r1.json()
            md = r1['text']
            print(f"[🌐 Visit-ES] Fetched {len(md)} characters from webpage")
        except Exception as e:
            erroutput = str(e)
            print(f"[❌ Visit-ES] Failed to fetch page: {erroutput}")
            logger.error(f"error: {erroutput}")
            logger.error(f"[Visit] Failed, url={url}")
            return "[Visit] Failed to read page.", 0.0, {}

        content = md
        useful_information = f"The content of {url} is as follows:\n\n{content}"
        print(f"[✅ Visit-ES] Visit completed, returned {len(useful_information)} characters")
        
        self._instance_dict[instance_id]["current_result"] = useful_information.strip()

        reward = 0.0

        metrics = {
            "query": goal,
            "response": useful_information,
            "operation_count": self._instance_dict[instance_id]["operation_count"],
        }

        return useful_information.strip(), reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate tool reward."""
        return 0.0

