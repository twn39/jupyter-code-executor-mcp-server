import os
import click
import logging
from pathlib import Path
from datetime import datetime
import asyncio
from typing import Optional, Literal, cast, Dict
from mcp.server import FastMCP
from jupyter_code_executor_mcp_server.prompts import data_analyst_prompt
from autogen_core import CancellationToken
from autogen_core.code_executor import CodeBlock
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from jupyter_client.kernelspec import KernelSpecManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("JupyterMCP")

# Global session storage
_sessions: Dict[str, JupyterCodeExecutor] = {}

async def server_lifespan(context):
    yield
    # Cleanup all sessions on shutdown
    logger.info(f"Cleaning up {len(_sessions)} active sessions...")
    for session_id, executor in _sessions.items():
        try:
            await executor.stop()
            logger.info(f"Stopped session {session_id}")
        except Exception as e:
            logger.error(f"Failed to stop session {session_id}: {e}")
    _sessions.clear()

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
        if session_id in _sessions:
            executor = _sessions[session_id]
        else:
            # Create new persistent session
            logger.info(f"Creating new session: {session_id} with kernel {kernel_name}")
            executor = JupyterCodeExecutor(
                kernel_name=kernel_name,
                output_dir=output_dir,
                timeout=int(os.environ.get("SESSION_TIMEOUT", 600))
            )
            await executor.start()
            _sessions[session_id] = executor
    else:
        # Create temporary executor
        is_temporary = True
        executor = JupyterCodeExecutor(
            kernel_name=kernel_name,
            output_dir=output_dir,
            timeout=int(os.environ.get("SESSION_TIMEOUT", 600))
        )
        await executor.start()

    try:
        cancellation_token = CancellationToken()
        code_block = CodeBlock(code=code, language="python") # Language is mainly for highlighting, kernel determines execution
        
        result = await executor.execute_code_blocks([code_block], cancellation_token)
        
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

    finally:
        if is_temporary:
            await executor.stop()

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
    
    mcp.settings.port = port or 5010
    _transport = cast(Literal["stdio", "sse", "streamable-http"], transport or "streamable-http")
    mcp.run(transport=_transport)


if __name__ == "__main__":
    try:
        serve()
    except KeyboardInterrupt:
        pass
