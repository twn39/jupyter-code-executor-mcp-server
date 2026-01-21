import time
import signal
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from autogen_ext.code_executors.jupyter import JupyterCodeExecutor

logger = logging.getLogger("JupyterMCP")


@dataclass
class SessionInfo:
    executor: JupyterCodeExecutor
    last_accessed: float


class SessionManager:
    """
    管理 Jupyter Kernel 会话的类。
    
    提供线程安全的会话创建、获取、清理功能，支持可配置的超时时间。
    """
    
    def __init__(self, timeout: int = 600, cleanup_interval: int = 60):
        """
        初始化 SessionManager。
        
        Args:
            timeout: 会话超时时间（秒），超过此时间未访问的会话将被自动清理。
            cleanup_interval: 后台清理任务的检查间隔（秒）。
        """
        self._sessions: dict[str, SessionInfo] = {}
        self._lock = asyncio.Lock()
        self._timeout = timeout
        self._cleanup_interval = cleanup_interval
        self._cleanup_task: Optional[asyncio.Task[None]] = None
    
    @property
    def session_count(self) -> int:
        """当前活跃会话数量。"""
        return len(self._sessions)
    
    async def get_or_create(
        self,
        session_id: str,
        kernel_name: str,
        output_dir: Path
    ) -> JupyterCodeExecutor:
        """
        获取或创建一个会话。
        
        如果指定 session_id 的会话已存在，则返回现有的 executor 并更新访问时间。
        否则，创建一个新的 executor 并存储。
        
        Args:
            session_id: 会话唯一标识符。
            kernel_name: Jupyter Kernel 名称。
            output_dir: 代码输出目录。
        
        Returns:
            JupyterCodeExecutor 实例。
        """
        async with self._lock:
            if session_id not in self._sessions:
                executor = JupyterCodeExecutor(
                    kernel_name=kernel_name,
                    output_dir=output_dir,
                    timeout=self._timeout
                )
                original_sigint = signal.getsignal(signal.SIGINT)
                await executor.start()
                signal.signal(signal.SIGINT, original_sigint)
                self._sessions[session_id] = SessionInfo(executor, time.time())
                logger.info(f"创建新会话: {session_id} (kernel: {kernel_name})")
            
            info = self._sessions[session_id]
            info.last_accessed = time.time()
            return info.executor
    
    async def close_session(self, session_id: str) -> None:
        """
        安全关闭指定会话。
        
        Args:
            session_id: 要关闭的会话 ID。
        """
        async with self._lock:
            if session_id in self._sessions:
                info = self._sessions.pop(session_id)
                logger.info(f"正在关闭会话: {session_id}")
                try:
                    original_sigint = signal.getsignal(signal.SIGINT)
                    await asyncio.shield(info.executor.stop())
                    signal.signal(signal.SIGINT, original_sigint)
                except (AssertionError, Exception, asyncio.CancelledError, GeneratorExit) as e:
                    logger.warning(f"关闭会话 {session_id} 时发生错误 (已忽略): {e}")
    
    async def cleanup_expired(self) -> None:
        """清理所有过期的会话。"""
        now = time.time()
        expired_ids: list[str] = []
        
        async with self._lock:
            expired_ids = [
                sid for sid, info in self._sessions.items()
                if now - info.last_accessed > self._timeout
            ]
        
        for sid in expired_ids:
            logger.info(f"清理过期会话: {sid}")
            await self.close_session(sid)
    
    async def close_all(self) -> None:
        """关闭所有活跃会话。"""
        if not self._sessions:
            return
        
        logger.info(f"正在关闭所有活跃会话 ({len(self._sessions)} 个)...")
        
        async with self._lock:
            session_ids = list(self._sessions.keys())
        
        tasks = [asyncio.create_task(self.close_session(sid)) for sid in session_ids]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def start_cleanup_loop(self) -> None:
        """启动后台清理任务。"""
        if self._cleanup_task is None or self._cleanup_task.done():
            logger.info("启动后台清理任务...")
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
    
    async def stop_cleanup_loop(self) -> None:
        """停止后台清理任务。"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("后台清理任务已停止")
    
    async def _cleanup_loop(self) -> None:
        """后台清理循环。"""
        logger.info("后台清理任务已启动...")
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval)
                await self.cleanup_expired()
        except asyncio.CancelledError:
            logger.info("后台清理任务收到停止信号")
