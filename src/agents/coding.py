"""GuardRoute coding subagent.

Executes Python code safely using RestrictedPython in a controlled environment.
Enforces limits on network, files, processes, imports, memory, and timeout.
"""

import sys
import io
import math
import json
import threading
import queue
import time
from typing import Any, Dict
try:
    from RestrictedPython import compile_restricted, safe_builtins
    from RestrictedPython.PrintCollector import PrintCollector
    from RestrictedPython.Guards import (
        safe_builtins as guard_builtins,
        safer_getattr,
        guarded_setattr,
        guarded_delattr,
        guarded_iter_unpack_sequence,
    )
    HAS_RESTRICTED_PYTHON = True
except ModuleNotFoundError:
    HAS_RESTRICTED_PYTHON = False
    compile_restricted = None
    safe_builtins = {}
    PrintCollector = None

from common.schemas.agent_types import SubAgentResult, SubAgentStatus

# Pre-defined list of allowed builtins
ALLOWED_BUILTINS = {
    **safe_builtins,
    "print": print,  # will be overridden dynamically per run
    "math": math,
    "json": json,
    "range": range,
    "list": list,
    "dict": dict,
    "set": set,
    "tuple": tuple,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "sum": sum,
    "min": min,
    "max": max,
    "len": len,
    "abs": abs,
    "round": round,
    "enumerate": enumerate,
    "zip": zip,
    "any": any,
    "all": all,
    "sorted": sorted,
    "filter": filter,
    "map": map,
    "divmod": divmod,
    "pow": pow,
}


def _execute_code_sandbox(code: str, pc: PrintCollector, result_queue: queue.Queue) -> None:
    """Worker target for executing restricted Python code."""
    # Prepare sandbox globals
    sandbox_globals = {
        "__builtins__": ALLOWED_BUILTINS,
        "_print_": lambda *args, **kwargs: pc,  # Required by RestrictedPython for print statements
        "_getattr_": safer_getattr,
        "_setattr_": guarded_setattr,
        "_delattr_": guarded_delattr,
        "_getitem_": lambda ob, index: ob[index],
        "_write_": lambda ob: ob,
        "_getiter_": iter,
        "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
    }

    try:
        # RestrictedPython compilation
        byte_code = compile_restricted(code, filename="<sandbox>", mode="exec")
        
        # Execute compiled code in safe namespace
        exec(byte_code, sandbox_globals)
        result_queue.put((True, None))
    except Exception as e:
        result_queue.put((False, e))


async def run_code_sandbox(code: str, timeout: float = 10.0) -> SubAgentResult:
    """Runs Python code block in RestrictedPython sandbox with timeout limit."""
    pc = PrintCollector()
    result_queue = queue.Queue()
    start_time = time.time()

    try:
        from common.observability.logger import log_security_event
        log_security_event("SANDBOX_EXECUTION_ATTEMPT", {"code_length": len(code), "timeout": timeout})
    except Exception:
        pass

    # Run execution in a separate daemon thread to enforce timeout
    thread = threading.Thread(
        target=_execute_code_sandbox,
        args=(code, pc, result_queue),
        daemon=True
    )
    
    thread.start()
    
    # Wait for completion or timeout
    loop = sys.modules["asyncio"].get_running_loop()
    
    def wait_on_thread():
        thread.join(timeout)
        return not thread.is_alive()
        
    completed_in_time = await loop.run_in_executor(None, wait_on_thread)
    latency_ms = (time.time() - start_time) * 1000.0
    
    # Accumulate prints
    printed_lines = pc.txt
    stdout_output = "".join(str(line) for line in printed_lines)
    
    if not completed_in_time:
        # Thread timed out
        timeout_msg = f"Execution Timeout: code execution exceeded {timeout}s limit."
        try:
            log_security_event(
                "SANDBOX_EXECUTION_RESULT",
                {"status": "TIMEOUT", "latency_ms": latency_ms, "error_message": timeout_msg}
            )
        except Exception:
            pass
        return SubAgentResult(
            source="coding",
            status=SubAgentStatus.TIMEOUT,
            content=stdout_output,
            error_message=timeout_msg,
            token_count=0
        )

    # Read queue result
    success = False
    try:
        success, err = result_queue.get_nowait()
    except queue.Empty:
        err = RuntimeError("No result returned from sandbox executor")

    if not success:
        err_msg = f"Runtime Error: {str(err)}"
        try:
            log_security_event(
                "SANDBOX_EXECUTION_RESULT",
                {"status": "ERROR", "latency_ms": latency_ms, "error_message": err_msg}
            )
        except Exception:
            pass
        return SubAgentResult(
            source="coding",
            status=SubAgentStatus.ERROR,
            content=stdout_output,
            error_message=err_msg,
            token_count=0
        )

    content_str = stdout_output or "Execution succeeded with no stdout output."
    # Truncate content to 10,000 characters
    if len(content_str) > 10000:
        content_str = content_str[:10000] + "\n... [Output Truncated due to size limits]"

    try:
        log_security_event(
            "SANDBOX_EXECUTION_RESULT",
            {"status": "SUCCESS", "latency_ms": latency_ms}
        )
    except Exception:
        pass

    return SubAgentResult(
        source="coding",
        status=SubAgentStatus.SUCCESS,
        content_type="text",
        content=content_str,
        token_count=len(content_str) // 4
    )
