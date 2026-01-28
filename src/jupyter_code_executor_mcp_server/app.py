import os
import click
import logging
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Optional, Dict
from mcp.server import FastMCP
from jupyter_code_executor_mcp_server.prompts import data_analyst_prompt
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from jupyter_code_executor_mcp_server.safe_notebook import SafeJupyterCodeExecutor
from jupyter_client.kernelspec import KernelSpecManager
import inspect

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("JupyterMCP")

# Global session storage
# Global session storage
class SessionManager:
    def __init__(self):
        self._sessions: Dict[str, SafeJupyterCodeExecutor] = {}
        self._last_accessed: Dict[str, float] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    async def get_or_create(self, session_id: str, kernel_name: str, output_dir: Path) -> SafeJupyterCodeExecutor:
        # Update access time
        self._last_accessed[session_id] = datetime.now().timestamp()
        
        if session_id in self._sessions:
            return self._sessions[session_id]
        
        logger.info(f"Creating new session: {session_id} with kernel {kernel_name}")
        executor = SafeJupyterCodeExecutor(
            kernel_name=kernel_name,
            output_dir=output_dir,
            timeout=int(os.environ.get("SESSION_TIMEOUT", 600))
        )
        await executor.start()
        self._sessions[session_id] = executor
        return executor
    
    async def stop(self, session_id: str):
        if session_id in self._sessions:
            executor = self._sessions[session_id]
            try:
                # Shield execution to prevent cancellation of cleanup
                # Wait for 2 seconds
                await asyncio.wait_for(asyncio.shield(executor.stop()), timeout=2.0)
                logger.info(f"Stopped session {session_id}")
            except asyncio.TimeoutError:
                logger.warning(f"Timeout while stopping session {session_id}")
            except Exception as e:
                logger.error(f"Error stopping session {session_id}: {e}")
            finally:
                # Force remove even if stop failed
                if session_id in self._sessions:
                    del self._sessions[session_id]
                if session_id in self._last_accessed:
                    del self._last_accessed[session_id]

    async def cleanup_loop(self):
        timeout_seconds = int(os.environ.get("SESSION_TIMEOUT", 600))
        check_interval = 60 # Check every minute
        
        logger.info(f"Starting session cleanup loop (timeout={timeout_seconds}s)")
        
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(check_interval)
                now = datetime.now().timestamp()
                
                # Identify expired sessions
                expired_sessions = []
                for session_id, last_access in list(self._last_accessed.items()):
                    if now - last_access > timeout_seconds:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    logger.info(f"Session {session_id} expired (idle > {timeout_seconds}s). Cleaning up.")
                    await self.stop(session_id)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def start_cleanup(self):
        self._shutdown_event.clear()
        self._cleanup_task = asyncio.create_task(self.cleanup_loop())

    async def shutdown_all(self):
        self._shutdown_event.set()
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Cleaning up {len(self._sessions)} active sessions...")
        # Create a copy of keys to iterate safely
        session_ids = list(self._sessions.keys())
        # Run all stops concurrently to speed up shutdown
        tasks = [self.stop(sid) for sid in session_ids]
        if tasks:
             await asyncio.gather(*tasks)
        self._sessions.clear()

session_manager = SessionManager()

from contextlib import asynccontextmanager

@asynccontextmanager
async def server_lifespan(server):
    session_manager.start_cleanup()
    yield
    await session_manager.shutdown_all()
    
    # Debug: Check for hanging tasks
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    if tasks:
        logger.info(f"Found {len(tasks)} pending tasks on shutdown:")
        for t in tasks:
            logger.info(f"  - {t.get_name()}: {t.get_coro()}")

mcp = FastMCP(
    "Jupyter Code Executor MCP Server",
    stateless_http=False,
    json_response=False,
    streamable_http_path="/mcp",
    lifespan=server_lifespan,
)


@mcp.tool()
def list_kernels() -> str:
    """
    列出当前环境所有已安装并可用的 Jupyter Kernels。

    Returns:
        返回一个字符串列表，包含 Kernel 的内部名称（用于调用）和显示名称。
    """
    ksm = KernelSpecManager()
    specs = ksm.get_all_specs()
    
    result = []
    for name, spec in specs.items():
        display_name = spec.get('spec', {}).get('display_name', name)
        result.append(f"- {name}: {display_name}")
    
    return "\n".join(result)


@mcp.prompt()
def list_prompts() -> str:
    """data analyst prompt"""
    return data_analyst_prompt


@mcp.tool()
async def execute_code(code: str, kernel_name: str, session_id: Optional[str] = None) -> str:
    """
    使用指定的 Jupyter Kernel 执行代码片段。
    支持多语言(Python, R, Julia 等），具体取决于系统安装的 Kernel。

    Args:
        code: 需要执行的代码字符串。
        kernel_name: 目标 Kernel 的名称（例如 'python3' 或 'ir'）。
                     请务必使用 list_kernels 工具获取准确的名称。
        session_id: (可选) 会话 UUID。如果提供, 将尝试复用由于该 ID 关联的 Kernel 上下文。
                    这允许在同一次对话中保持变量状态。如果不提供，将创建临时的无状态执行环境。
    """
    output_dir = Path(os.getenv("OUTPUT_DIR", "~/output")).expanduser()
    
    # Determine which executor to use
    executor = None
    is_temporary = False

    if session_id:
        executor = await session_manager.get_or_create(session_id, kernel_name, output_dir)
    else:
        # Create temporary executor
        is_temporary = True
        executor = SafeJupyterCodeExecutor(
            kernel_name=kernel_name,
            output_dir=output_dir,
            timeout=int(os.environ.get("SESSION_TIMEOUT", 600))
        )
        await executor.start()

    try:
        cancellation_token = CancellationToken()
        code_block = CodeBlock(code=code, language="python") # Language is mainly for highlighting, kernel determines execution
        
        logger.info(f"Executing code block in session {session_id or 'temp'}")
        result = await executor.execute_code_blocks([code_block], cancellation_token)
        logger.info("Code execution completed")
        
        response = []
        if result.output:
            response.append(f"Output:\n{result.output}")
        if result.output_files:
            response.append("Generated Files:")
            for f in result.output_files:
                response.append(f"- {f}")
        
        if result.exit_code != 0:
            response.append(f"\nExecution failed with exit code {result.exit_code}")

        return "\n".join(response)

    except asyncio.CancelledError:
        logger.info("Execution cancelled by system (CancelledError)")
        cancellation_token.cancel()
        logger.info("Cancellation token triggered, re-raising CancelledError")
        raise
    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        if is_temporary:
            logger.info("Stopping temporary session")
            await executor.stop()
            logger.info("Temporary session stopped")

@mcp.tool()
def list_files() -> str:
    """
    List all files or directories in user workspace

    Returns:
        文件列表
    """
    user_dir = Path(os.getenv('DATA_DIR') or "~/data").expanduser()
    if not user_dir.exists():
        user_dir.mkdir(parents=True, exist_ok=True)

    all_items = [f.resolve() for f in user_dir.iterdir()]
    return "\n".join([str(item) for item in all_items])


@mcp.tool()
def get_current_time() -> str:
    """
    Get the current time in a more human-readable format.
    """
    now = datetime.now()
    current_time = now.strftime("%I:%M:%S %p")
    current_date = now.strftime("%A, %B %d, %Y")

    return f"Current Date and Time = {current_date}, {current_time}"


@click.command()
@click.option("--transport", "-t", type=str, help="server transport, default: streamable-http")
@click.option("--port", "-p", type=int, help="server listening port, default: 5010")
@click.option("--data_dir", "-d", type=str, help="user data dir, default: ~/data")
@click.option("--output_dir", "-o", type=str, help="code output dir, default: ~/output")
@click.option("--session-timeout", "-st", type=int, help="session timeout in seconds, default: 600")
def serve(
    transport: Optional[str],
    port: Optional[int],
    data_dir: Optional[str],
    output_dir: Optional[str],
    session_timeout: Optional[int]
):
    os.environ["DATA_DIR"] = data_dir or "~/data"
    os.environ["OUTPUT_DIR"] = output_dir or "~/output"
    os.environ["SESSION_TIMEOUT"] = str(session_timeout or 600)
    
    os.environ["DATA_DIR"] = data_dir or "~/data"
    os.environ["OUTPUT_DIR"] = output_dir or "~/output"
    os.environ["SESSION_TIMEOUT"] = str(session_timeout or 600)
    
    port = port or 5010
    
    # We enforce streamable-http and run uvicorn manually to control shutdown timeout
    # mcp.run(transport="streamable-http") 
    
    import uvicorn
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response

    # Middleware to suppress CancelledError during shutdown to avoid noisy tracebacks
    async def suppress_cancellation_middleware(scope, receive, send):
        try:
            # streamable_http_app is a method that returns the ASGI app
            app = mcp.streamable_http_app
            if callable(app) and not inspect.iscoroutinefunction(app) and not isinstance(app, type):
                 # It's a method returning the app
                 app = app()
            
            await app(scope, receive, send)
        except asyncio.CancelledError:
            pass
        except Exception:
            raise

    logger.info(f"Starting server on port {port} with graceful shutdown timeout of 2s")
    
    uvicorn.run(
        suppress_cancellation_middleware,
        host="0.0.0.0", 
        port=port, 
        timeout_graceful_shutdown=2.0
    )


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        pass
