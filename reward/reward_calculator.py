import re
import json
import os
from typing import Optional
import httpx
import time
from openai import OpenAI

from reward.base_reward_calculator import BaseRewardCalculator


# Initialize OpenAI client for LLM-based reward calculation
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    http_client=httpx.Client(verify=False)
)


class GoogleandPythonRewardCalculator(BaseRewardCalculator):
    """Reward calculator using LLM-based semantic matching."""

    @staticmethod
    def __llm_judge(pred: str, gold: str) -> float:
        """Content reward, call OpenAI GPT to judge if pred and gold are semantically consistent."""
        prompt = (
            f"Predicted Answer: {pred}\n"
            f"Expected Answer: {gold}\n"
            "Determine if the predicted answer is semantically correct (case-insensitive). "
            'Reply **only** a JSON object in the format: {"correct": 0} or {"correct": 1} '
            "Do not include any additional text or markdown formatting."
        )
        try:
            t0_sum = time.perf_counter()
            resp = openai_client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-oss"),
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1024,
                temperature=0
            )
            t1_sum = time.perf_counter()
            print(f"LLM JUDGE took {t1_sum-t0_sum:.3f}s")
            
            raw = resp.choices[0].message.content.strip()
            print("[DEBUG] LLM raw return:", raw)
            
            # Handle special formatting
            raw = re.split(r'<\|start\|>assistant<\|channel\|>final<\|message\|>', raw)[-1]
            text = raw.split(r'<\|end\|>')[0]
            text = text.strip()
            text = re.sub(r'^```json\s*', '', text, flags=re.I)
            text = re.sub(r'^```\s*', '', text)
            text = re.sub(r'\s*```\s*$', '', text)
            data = json.loads(text)["correct"]
            return float(data)

        except Exception as e:
            print(f"[DEBUG] LLM judge error: {e}")
            return 0.0

    @staticmethod
    def extract_output(output_str: str) -> str:
        """Extract output from the response string."""
        return output_str

    @classmethod
    def _verify_correction(cls, extracted_output, identity: dict) -> float:
        """Verify correctness by extracting answer and comparing with ground truth."""
        matches = re.findall(r"<ANSWER>\s*(.*?)\s*</ANSWER>", extracted_output, flags=re.DOTALL)
        pred = matches[-1].strip() if matches else ""
        # If pred returns "answer", it means it matched the prompt from the end, i.e., didn't reach the final step
        expected = str(identity)

        print("[REWARD] identity:", identity)
        print("[REWARD] pred:", pred)

        # Answer score
        a_score = cls.__llm_judge(pred, expected) if pred else 0.0
        total_reward = a_score
        print("[REWARD] total_reward:", total_reward)
        return total_reward

