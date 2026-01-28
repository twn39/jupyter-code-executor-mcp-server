import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from jupyter_code_executor_mcp_server.safe_notebook import SafeJupyterCodeExecutor

logger = logging.getLogger("JupyterMCP")

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
                
                logger.info(f"Active sessions: {len(self._sessions)}")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def start_cleanup(self):
        if self._cleanup_task and not self._cleanup_task.done():
            logger.warning("Cleanup task is already running. Ignoring start_cleanup request.")
            return
            
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
