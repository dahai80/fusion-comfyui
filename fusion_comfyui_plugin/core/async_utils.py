import asyncio
import concurrent.futures
import logging
import threading

logger = logging.getLogger("fusion_comfyui.async_utils")

_shared_executor = None
_executor_lock = threading.Lock()


def get_shared_executor():
    global _shared_executor
    if _shared_executor is None:
        with _executor_lock:
            if _shared_executor is None:
                _shared_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="fusion_async"
                )
                logger.info("async_utils: created shared ThreadPoolExecutor")
    return _shared_executor


def run_async(coro, timeout=600):
    executor = get_shared_executor()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        future = executor.submit(asyncio.run, coro)
        return future.result(timeout=timeout)
    else:
        return asyncio.run(coro)
