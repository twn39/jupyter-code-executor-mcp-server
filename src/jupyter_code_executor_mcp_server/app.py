import os
import click
import signal
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal, cast
from mcp.server import FastMCP
from autogen_core import CancellationToken
from contextlib import asynccontextmanager
from autogen_core.code_executor import CodeBlock
from jupyter_client.kernelspec import KernelSpecManager
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from jupyter_code_executor_mcp_server.prompts import data_analyst_prompt
from jupyter_code_executor_mcp_server.session import SessionManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("JupyterMCP")

# 全局 SessionManager 实例和连接计数
_session_manager: Optional[SessionManager] = None
_active_connections: int = 0


def get_session_manager() -> SessionManager:
    """获取全局 SessionManager 实例。"""
    global _session_manager
    if _session_manager is None:
        timeout = int(os.getenv("SESSION_TIMEOUT") or "600")
        _session_manager = SessionManager(timeout=timeout)
    return _session_manager


@asynccontextmanager
async def server_lifespan(server: FastMCP):
    global _active_connections

    session_manager = get_session_manager()
    _active_connections += 1

    # 仅在第一个连接建立时启动清理任务
    if _active_connections == 1:
        await session_manager.start_cleanup_loop()
    else:
        logger.info(f"复用现有清理任务 (当前活跃连接数: {_active_connections})")

    try:
        yield
    finally:
        _active_connections -= 1
        logger.info(f"连接断开 (剩余活跃连接数: {_active_connections})")

        # 仅当没有活跃连接时，才执行清理和关闭操作
        if _active_connections <= 0:
            logger.info("无活跃连接，停止后台清理任务并关闭所有会话...")
            await session_manager.stop_cleanup_loop()
            await session_manager.close_all()


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
    try:
        ksm = KernelSpecManager()
        specs = ksm.get_all_specs()

        if not specs:
            return "No Jupyter kernels found on this system."

        output = "Available Jupyter Kernels:\n"
        output += "Format: [Kernel Name] - [Display Name]\n"
        output += "-" * 40 + "\n"

        for name, details in specs.items():
            display_name = details.get('spec', {}).get('display_name', 'Unknown')
            output += f"- {name}: {display_name}\n"

        return output

    except Exception as e:
        return f"Error listing kernels: {str(e)}"


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
    try:
        output_dir = Path(os.getenv('OUTPUT_DIR') or "~/output").expanduser()
        session_manager = get_session_manager()

        if session_id is not None:
            # 使用 SessionManager 获取或创建会话
            executor = await session_manager.get_or_create(
                session_id=session_id,
                kernel_name=kernel_name,
                output_dir=output_dir
            )
            
            cancel_token = CancellationToken()
            code_blocks = [CodeBlock(code=code, language=kernel_name)]
            result = await executor.execute_code_blocks(code_blocks, cancel_token)
            
            output_msg = f"Kernel: {kernel_name} (Session: {session_id})\n"
            output_msg += f"Exit Code: {result.exit_code}\n"
            output_msg += f"Output:\n{result.output}"
            return output_msg

        else:
            # Stateless execution (maintain backward compatibility)
            executor = JupyterCodeExecutor(kernel_name=kernel_name, timeout=600, output_dir=output_dir)
            original_sigint = signal.getsignal(signal.SIGINT)
            try:
                await executor.start()
                signal.signal(signal.SIGINT, original_sigint)

                cancel_token = CancellationToken()
                code_blocks = [CodeBlock(code=code, language=kernel_name)]
                result = await executor.execute_code_blocks(code_blocks, cancel_token)
            finally:
                try:
                    await executor.stop()
                except (AssertionError, Exception):
                    pass
                signal.signal(signal.SIGINT, original_sigint)

                output_msg = f"Kernel: {kernel_name}\n"
                output_msg += f"Exit Code: {result.exit_code}\n"
                output_msg += f"Output:\n{result.output}"

                return output_msg

    except Exception as e:
        return f"Error executing code with kernel '{kernel_name}': {str(e)}"


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
