import asyncio
import atexit
import signal
import typing as t
from contextlib import asynccontextmanager
from time import monotonic

from nbclient import NotebookClient
from nbclient.util import run_hook, ensure_async

from autogen_ext.code_executors.jupyter import JupyterCodeExecutor
from nbformat import NotebookNode
from nbformat import v4 as nbformat

class SafeNotebookClient(NotebookClient):
    """
    A subclass of NotebookClient that:
    1. Does NOT register signal handlers (SIGINT) which conflicts with Uvicorn/FastMCP.
    2. Makes _async_cleanup_kernel idempotent to prevent AssertionErrors.
    """

    async def _async_cleanup_kernel(self) -> None:
        # Idempotency check: if km is None, we are already cleaned up.
        if self.km is None:
            return

        # Original assertion from nbclient
        # assert self.km is not None 
        
        # We proceed with cleanup
        # We call super but we need to be careful if super raises assertion.
        # Since we checked self.km is not None, super call should be safe IF it doesn't do other checks.
        # But looking at source, it does logic.
        # To be purely safe and avoid copy-pasting the WHOLE cleanup logic which relies on other private attributes,
        # we can try to call super.
        # However, super._async_cleanup_kernel sets self.km = None at the end.
        
        try:
            await super()._async_cleanup_kernel()
        except AssertionError:
            # If it still fails for some reason, we swallow it because we are cleaning up anyway
            pass
        except Exception:
            raise

    @asynccontextmanager
    async def async_setup_kernel(self, **kwargs: t.Any) -> t.AsyncGenerator[None, None]:
        """
        Modified version of async_setup_kernel that does NOT register signal handlers.
        """
        # by default, cleanup the kernel client if we own the kernel manager
        # and keep it alive if we don't
        cleanup_kc = kwargs.pop("cleanup_kc", self.owns_km)
        if self.km is None:
            self.km = self.create_kernel_manager()

        # self._cleanup_kernel uses run_async, which ensures the ioloop is running again.
        # This is necessary as the ioloop has stopped once atexit fires.
        atexit.register(self._cleanup_kernel)

        # DISABLED SIGNAL HANDLER REGISTRATION
        # def on_signal() -> None: ...
        # loop.add_signal_handler(...)

        if not self.km.has_kernel:
            await self.async_start_new_kernel(**kwargs)

        if self.kc is None:
            await self.async_start_new_kernel_client()

        try:
            yield
        except RuntimeError as e:
            await run_hook(self.on_notebook_error, notebook=self.nb)
            raise e
        finally:
            if cleanup_kc:
                await self._async_cleanup_kernel()
            await run_hook(self.on_notebook_complete, notebook=self.nb)
            atexit.unregister(self._cleanup_kernel)
            # No need to remove signal handlers we didn't add


class SafeJupyterCodeExecutor(JupyterCodeExecutor):
    """
    JupyterCodeExecutor that uses SafeNotebookClient.
    """
    async def start(self) -> None:
        """(Experimental) Start the code executor.

        Initializes the Jupyter Notebook execution environment by creating a new notebook and setting it up with the specified Jupyter Kernel.
        Marks the executor as started, allowing for code execution.
        This method should be called before executing any code blocks.
        """
        if self._started:
            return

        notebook: NotebookNode = nbformat.new_notebook()  # type: ignore

        # USE SafeNotebookClient
        self._client = SafeNotebookClient(
            nb=notebook,
            kernel_name=self._kernel_name,
            timeout=self._timeout,
            allow_errors=True,
            shutdown_kernel="immediate", # Force kill on shutdown for speed
        )

        self.kernel_context = self._client.async_setup_kernel()
        await self.kernel_context.__aenter__()

        self._started = True
