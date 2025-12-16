import logging
import os
import click
import time
import asyncio
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
from dataclasses import dataclass
from mcp.server import FastMCP
from autogen_core import CancellationToken
from contextlib import asynccontextmanager
from autogen_core.code_executor import CodeBlock
from jupyter_client.kernelspec import KernelSpecManager
from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from jupyter_code_executor_mcp_server.prompts import data_analyst_prompt

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("JupyterMCP")


@dataclass
class SessionInfo:
    executor: JupyterCodeExecutor
    last_accessed: float

sessions: Dict[str, SessionInfo] = {}

async def _cleanup_loop(timeout: int):
    logger.info("后台清理任务已启动...")
    try:
        while True:
            await asyncio.sleep(60)
            now = time.time()
            expired_ids = [
                sid for sid, info in sessions.items()
                if now - info.last_accessed > timeout
            ]

            for sid in expired_ids:
                logger.info(f"清理过期会话: {sid}")
                await _close_session(sid)

    except asyncio.CancelledError:
        logger.info("后台清理任务收到停止信号")
        # 这里不需要做额外处理，直接退出循环即可，清理工作交给 lifespan 的 finally

async def _close_session(session_id: str):
    """安全关闭单个会话"""
    if session_id in sessions:
        info = sessions.pop(session_id)
        logger.info(f"正在关闭会话: {session_id}")
        try:
            # 关键修复：使用 shield 保护关闭过程，防止在关闭中途被再次取消导致状态不一致
            # Use stop() which includes checks, and catch AssertionError common in dry cleanups
            original_sigint = signal.getsignal(signal.SIGINT)
            # JupyterCodeExecutor.stop() is safer than __aexit__
            await asyncio.shield(info.executor.stop())
            signal.signal(signal.SIGINT, original_sigint)
        except (AssertionError, Exception, asyncio.exceptions.CancelledError) as e:
            # Capture AssertionError from nbclient double-cleanup
            logger.warning(f"关闭会话 {session_id} 时发生错误 (已忽略): {e}")

async def _shutdown_all_sessions():
    """关闭所有活跃会话"""
    if not sessions:
        return
    logger.info(f"正在关闭所有活跃会话 ({len(sessions)} 个)...")
    # 创建副本进行遍历
    active_sessions = list(sessions.keys())
    # 并发关闭可以加快退出速度
    tasks = [asyncio.create_task(_close_session(sid)) for sid in active_sessions]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)



# 新增全局变量来跟踪任务和连接数
_global_cleanup_task: Optional[asyncio.Task] = None
_active_connections: int = 0

@asynccontextmanager
async def server_lifespan(server: FastMCP):
    global _global_cleanup_task, _active_connections

    # 1. 连接计数 +1
    _active_connections += 1
    timeout = int(server.settings.debug) if False else 60

    # 2. 仅在第一个连接建立且任务未运行时，启动全局清理任务
    if _global_cleanup_task is None or _global_cleanup_task.done():
        logger.info("启动全局后台清理任务...")
        _global_cleanup_task = asyncio.create_task(_cleanup_loop(timeout))
    else:
        logger.info(f"复用现有清理任务 (当前活跃连接数: {_active_connections})")

    try:
        yield
    finally:
        # 3. 连接计数 -1
        _active_connections -= 1
        logger.info(f"连接断开 (剩余活跃连接数: {_active_connections})")

        # 4. 仅当没有活跃连接时，才执行清理和关闭操作
        if _active_connections <= 0:
            logger.info("无活跃连接，停止后台清理任务并关闭所有会话...")

            if _global_cleanup_task:
                _global_cleanup_task.cancel()
                try:
                    await _global_cleanup_task
                except asyncio.CancelledError:
                    pass
                _global_cleanup_task = None

            await _shutdown_all_sessions()


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
async def execute_code(code: str, kernel_name: str, session_id: str = None) -> str:
    """
    使用指定的 Jupyter Kernel 执行代码片段。
    支持多语言（Python, R, Julia 等），具体取决于系统安装的 Kernel。

    Args:
        code: 需要执行的代码字符串。
        kernel_name: 目标 Kernel 的名称（例如 'python3' 或 'ir'）。
                     请务必使用 list_kernels 工具获取准确的名称。
        session_id: (可选) 会话 ID。如果提供，将尝试复用由于该 ID 关联的 Kernel 上下文。
                    这允许在同一次对话中保持变量状态。如果不提供，将创建临时的无状态执行环境。
    """
    try:
        output_dir = Path(os.getenv('OUTPUT_DIR')).expanduser()

        if session_id is not None:
            if session_id not in sessions:
                executor = JupyterCodeExecutor(kernel_name=kernel_name, output_dir=output_dir, timeout=600)
                # Capture signal before enter
                original_sigint = signal.getsignal(signal.SIGINT)
                await executor.start()
                # Restore signal immediately after enter
                signal.signal(signal.SIGINT, original_sigint)
                sessions[session_id] = SessionInfo(executor, time.time())
            info = sessions[session_id]
            info.last_accessed = time.time() # 更新时间
        
            # Verify if kernel aligns (optional check, strictly we might assume session_id binds to a kernel type)
            # but for now we trust the session maps to the right executor or we could check executor.kernel_name if accessible
            cancel_token = CancellationToken()
            code_blocks = [CodeBlock(code=code, language=kernel_name)]
            result = await info.executor.execute_code_blocks(code_blocks, cancel_token)
            
            output_msg = f"Kernel: {kernel_name} (Session: {session_id})\n"
            output_msg += f"Exit Code: {result.exit_code}\n"
            output_msg += f"Output:\n{result.output}"
            return output_msg

        else:
            # Stateless execution (maintain backward compatibility)
            # 注意：JupyterCodeExecutor 每次都会启动新会话，保持无状态
            # 手动管理上下文以确保信号恢复
            executor = JupyterCodeExecutor(kernel_name=kernel_name, timeout=600, output_dir=output_dir)
            original_sigint = signal.getsignal(signal.SIGINT)
            try:
                await executor.start()
                signal.signal(signal.SIGINT, original_sigint) # Restore after start

                cancel_token = CancellationToken()
                code_blocks = [CodeBlock(code=code, language=kernel_name)]
                result = await executor.execute_code_blocks(code_blocks, cancel_token)
            finally:
                try:
                    await executor.stop()
                except (AssertionError, Exception):
                    pass # Ignore cleanup errors during shutdown
                signal.signal(signal.SIGINT, original_sigint) # Restore after exit

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
    user_dir = Path(os.getenv('DATA_DIR')).expanduser()
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
    current_time = now.strftime("%I:%M:%S %p")  # Using 12-hour format with AM/PM
    current_date = now.strftime(
        "%A, %B %d, %Y"
    )  # Full weekday, month name, day, and year

    return f"Current Date and Time = {current_date}, {current_time}"


@click.command()
@click.option("--transport", "-t", type=str, help="server transport, default: streamable-http")
@click.option("--port", "-p", type=int, help="server listening port, default: 5010")
@click.option("--data_dir", "-d", type=str, help="user data dir, default: ~/data")
@click.option("--output_dir", "-o", type=str, help="code output dir, default: ~/output")
@click.option("--session-timeout", "-st", type=int, help="session timeout in seconds, default: 600")
def serve(transport: Optional[str], port: Optional[int], data_dir: Optional[str], output_dir: Optional[str], session_timeout: Optional[int]):
    os.environ["DATA_DIR"] = data_dir or "~/data"
    os.environ["OUTPUT_DIR"] = output_dir or "~/output"
    
    mcp.settings.port = port or 5010
    _transport = transport or "streamable-http"
    mcp.run(transport=_transport)


if __name__ == "__main__":
    serve()
