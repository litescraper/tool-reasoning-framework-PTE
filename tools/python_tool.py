import logging
import os
import time
from typing import Any, Optional, Tuple
from uuid import uuid4

from tools.base_tool import BaseTool
from schemas.tool_schemas import OpenAIFunctionToolSchema
from core.rollout_trace import rollout_trace_op

from concurrent.futures import ThreadPoolExecutor

# Import sandbox_fusion - this should be available in the environment
try:
    from sandbox_fusion import (
        run_code, RunCodeRequest,
        run_concurrent,
        set_endpoint,
    )
except ImportError:
    # Fallback if sandbox_fusion is not available
    logger = logging.getLogger(__name__)
    logger.warning("sandbox_fusion not available. PythonTool will not work without it.")
    run_code = None
    RunCodeRequest = None
    run_concurrent = None
    set_endpoint = None

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

# Set sandbox endpoint
PYTHON_SERVICE_URL = os.getenv("PYTHON_SERVICE_URL", os.getenv("SANDBOX_URL", "http://localhost:8080"))
if set_endpoint:
    set_endpoint(PYTHON_SERVICE_URL)


class PythonTool(BaseTool):
    """Tool for executing Python code in a sandboxed environment."""
    
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._pool = ThreadPoolExecutor(max_workers=4)

    async def create(self, instance_id: Optional[str] = None, identity: dict = None, **kwargs) -> str:
        """Create a tool instance."""
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {
            "history": [],
            "current_result": "",
            "operation_count": 0
        }
        return instance_id

    async def _run_in_sandbox(self, code: str):
        """
        Execute code in sandbox using concurrent interface.
        """
        if not run_code or not RunCodeRequest or not run_concurrent:
            raise ImportError("sandbox_fusion is not available")
        
        req = RunCodeRequest(code=code, language="python", run_timeout=30)
        # run_concurrent returns List[Result], we only send one, take index 0
        results = run_concurrent(
            run_code,
            args=[[req]],  # Note: double nested list, each element is *args
        )
        logger.debug(f"[Python Tool] results: {results[0]}")
        return results[0]

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Python code in sandbox."""
        # 1. Parameter validation
        if "code" not in parameters:
            error_msg = "Missing parameter 'code'"
            print(f"[❌ Python] {error_msg}")
            logger.error("PythonTool execution error: missing parameter 'code'")
            return error_msg, 0.0, {}

        code = parameters["code"]
        if not isinstance(code, str):
            error_msg = "Parameter 'code' must be a string"
            print(f"[❌ Python] {error_msg}")
            logger.error("PythonTool execution error: 'code' must be a string")
            return error_msg, 0.0, {}
        
        code_preview = code[:100] + "..." if len(code) > 100 else code
        print(f"[🐍 Python] Executing code ({len(code)} chars): {code_preview}")
        print(f"[🐍 Python] Calling Python service: {PYTHON_SERVICE_URL}")
        
        # 2. Execute in remote sandbox
        try:
            t0 = time.perf_counter()
            code_result = await self._run_in_sandbox(code)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            print(f"[🐍 Python] Execution completed in {elapsed:.3f}s")
        except Exception as e:
            error_msg = f"Sandbox execution exception: {e}"
            print(f"[❌ Python] {error_msg}")
            logger.error(f"PythonTool sandbox execution exception: {e}")
            return error_msg, 0.0, {"error": str(e)}

        # 3. Assemble results
        stdout = code_result.run_result.stdout or ""
        stderr = code_result.run_result.stderr or ""
        if code_result.run_result.execution_time >= 29:  # Default 30s timeout
            stderr += "\n[Timeout] Execution time exceeded 30 seconds"
            print(f"[⚠️ Python] Execution timeout (>= 29s)")

        result_text = f"stdout:\n{stdout}\nstderr:\n{stderr}".strip()
        
        # Print execution results
        if stdout:
            stdout_preview = stdout[:200] + "..." if len(stdout) > 200 else stdout
            print(f"[🐍 Python] stdout ({len(stdout)} chars): {stdout_preview}")
        if stderr:
            stderr_preview = stderr[:200] + "..." if len(stderr) > 200 else stderr
            print(f"[⚠️ Python] stderr ({len(stderr)} chars): {stderr_preview}")
        if not stdout and not stderr:
            print(f"[🐍 Python] No output (empty stdout and stderr)")
        
        self._instance_dict[instance_id]["current_result"] = result_text
        self._instance_dict[instance_id]["operation_count"] += 1
        self._instance_dict[instance_id]["history"].append({
            "code": code,
            "stdout": stdout,
            "stderr": stderr
        })

        # 4. Response, reward, metrics
        response = result_text if result_text else "Finished execution."
        print(f"[✅ Python] Execution completed successfully")
        reward = 0.0  # Always 0
        metrics = {
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "operation_count": self._instance_dict[instance_id]["operation_count"]
        }
        return response, reward, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        """Calculate tool reward."""
        return 0.0

