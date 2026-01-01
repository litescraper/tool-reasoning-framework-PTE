import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4
import time

from tools.base_tool import BaseTool
from schemas.tool_schemas import OpenAIFunctionToolSchema
from core.rollout_trace import rollout_trace_op

import httpx
import requests
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class VisitTool(BaseTool):
    """Tool for visiting web pages and extracting relevant information."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        # Visit tool summary model configuration
        self.client = OpenAI(
            api_key=os.getenv("VISIT_SUMMARY_API_KEY", os.getenv("OPENAI_API_KEY", "EMPTY")),
            base_url=os.getenv("VISIT_SUMMARY_API_URL", os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1/")),
            http_client=httpx.Client(verify=False)
        )
        self.model_id = os.getenv("VISIT_SUMMARY_MODEL", os.getenv("OPENAI_MODEL", "gpt-oss"))
        self.jina_service_url = os.getenv("JINA_SERVICE_URL", "http://localhost:5003/jina")

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
        """Execute visit tool to extract information from a webpage."""
        try:
            url = parameters.get("url", "")
            goal = parameters.get("goal", "")
        except:
            return "[Visit] Invalid request format: Input must be a JSON object containing 'url' and 'goal' fields", 0.0, {}
        
        # Block Hugging Face URLs
        if self._is_huggingface_dataset_or_space_url(url):
            logger.warning(f"[Visit] Blocked HuggingFace URL, url={url}")
            return "[Visit] You are trying to visit a Hugging Face dataset for answers, please do not use the visit tool for this purpose.", 0.0, {}
        
        self._instance_dict[instance_id]["operation_count"] += 1
        print(f"[🌐 Visit] Starting visit to: {url}")
        print(f"[🌐 Visit] Goal: {goal}")
        
        try:
            t0 = time.perf_counter()
            print(f"[🌐 Visit] Calling Jina service: {self.jina_service_url}")
            r1 = requests.get(self.jina_service_url, params={'url': url}, timeout=30)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[🌐 Visit] Jina service completed in {elapsed:.3f}s")
            logger.info(f"[Visit] Success, took {elapsed:.3f}s, url={url}")
            
            md = r1.text
            print(f"[🌐 Visit] Fetched {len(md)} characters from webpage")
        except Exception as e:
            erroutput = str(e)
            print(f"[❌ Visit] Failed to fetch page: {erroutput}")
            logger.error(f"error: {erroutput}")
            logger.error(f"[Visit] Failed, url={url}")
            return "[Visit] Failed to read page.", 0.0, {}

        content = md[:150000]
        print(f"[🌐 Visit] Extracting information using summary model: {self.model_id}")
        EXTRACTOR_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rational**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" fields**
"""
        messages = [{"role": "user", "content": EXTRACTOR_PROMPT.format(webpage_content=content, goal=goal)}]
        raw = await self._extract_information(messages)
        
        try:
            # Try to parse JSON
            raw = json.loads(raw)
            useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            useful_information += "Evidence in page: \n" + str(raw["evidence"]) + "\n\n"
            useful_information += "Summary: \n" + str(raw["summary"]) + "\n\n"
            print(f"[✅ Visit] Successfully extracted information (JSON parsed)")
            logger.info("Parse JSON SUCCESSFUL!")
        except Exception as e:
            print(f"[⚠️ Visit] JSON parsing failed, using fallback format")
            logger.error(f"[ERROR] Parse JSON FAILED! error: {e}")
            logger.error(f"[ERROR] Parse JSON FAILED! raw: {raw}")
            useful_information = f"The useful information in {url} for user goal {goal} as follows: \n\n"
            useful_information += "Evidence in page: \n" + "The provided webpage content could not be accessed. Please check the URL or file format." + "\n\n"
            useful_information += "Summary: \n" + "The webpage content could not be processed, and therefore, no information is available." + "\n\n"
        
        # Update instance state
        self._instance_dict[instance_id]["current_result"] = useful_information.strip()
        print(f"[✅ Visit] Visit completed, extracted {len(useful_information)} characters")

        reward = 0.0

        metrics = {
            "query": goal,
            "response": useful_information,
            "operation_count": self._instance_dict[instance_id]["operation_count"],
        }

        return useful_information.strip(), reward, metrics

    async def _extract_information(self, msgs) -> str:
        """Extract information from webpage content using LLM."""
        try:
            t0ex = time.perf_counter()
            chat_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=msgs,
                temperature=0.7
            )
            t1ex = time.perf_counter()
            elapsed = t1ex - t0ex
            print(f"[🌐 Visit] Summary extraction completed in {elapsed:.3f}s")
            logger.info(f"Extract API call took {elapsed:.3f}s")
            
            content = chat_response.choices[0].message.content
            if content:
                try:
                    json.loads(content)
                except:
                    # Extract JSON from string
                    left = content.find('{')
                    right = content.rfind('}')
                    if left != -1 and right != -1 and left <= right:
                        content = content[left:right+1]
                    logger.warning(f"LOAD JSON FAILED! content converted to: {content}")
                return content
        except Exception as e:
            logger.error(f"[ERROR Extract] {e}")
            return "Extract information API failed."

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate tool reward."""
        return 0.0

