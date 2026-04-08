"""Claude CLI subprocess runner for market probability analysis."""

import asyncio
import time
from typing import Optional

import structlog

from src.config import settings
from src.llm.parser import LLMSignalOutput, parse_claude_output

logger = structlog.get_logger()


class ClaudeRunner:
    """Runs Claude CLI as a subprocess for market analysis.

    Uses asyncio.create_subprocess_exec (not shell) to safely invoke
    the claude binary with arguments passed as a list. The prompt is
    passed via the -p flag — no shell interpolation occurs.
    """

    def __init__(
        self,
        timeout: int | None = None,
        max_retries: int | None = None,
    ):
        self._timeout = timeout or settings.llm_timeout_seconds
        self._max_retries = max_retries or settings.llm_max_retries

    async def evaluate(self, prompt: str) -> Optional[LLMSignalOutput]:
        """Run Claude CLI with a prompt and parse the structured output.

        Returns parsed LLMSignalOutput or None on failure.
        """
        for attempt in range(self._max_retries + 1):
            start = time.monotonic()
            try:
                result = await self._run_cli(prompt)
                elapsed = time.monotonic() - start

                if result is None:
                    logger.warning(
                        "claude_empty_output",
                        attempt=attempt + 1,
                        elapsed=f"{elapsed:.1f}s",
                    )
                    continue

                parsed = parse_claude_output(result)
                if parsed:
                    logger.info(
                        "claude_eval_success",
                        probability=parsed.probability,
                        confidence=parsed.confidence,
                        elapsed=f"{elapsed:.1f}s",
                    )
                    return parsed

                logger.warning(
                    "claude_parse_failed",
                    attempt=attempt + 1,
                    output_preview=result[:200] if result else "",
                )

            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                logger.warning(
                    "claude_timeout",
                    attempt=attempt + 1,
                    timeout=self._timeout,
                    elapsed=f"{elapsed:.1f}s",
                )
            except Exception as e:
                elapsed = time.monotonic() - start
                logger.error(
                    "claude_error",
                    attempt=attempt + 1,
                    error=str(e),
                    elapsed=f"{elapsed:.1f}s",
                )

            if attempt < self._max_retries:
                await asyncio.sleep(2**attempt)

        return None

    async def _run_cli(self, prompt: str) -> Optional[str]:
        """Invoke claude CLI safely via create_subprocess_exec (no shell).

        Arguments are passed as a list — no shell interpolation, no
        injection risk. This is the Python equivalent of execFile.
        """
        cmd = [
            "claude",
            "-p", prompt,
            "--output-format", "json",
            "--model", settings.llm_model,
        ]

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._timeout,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise

        if proc.returncode != 0:
            err_msg = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
            logger.warning(
                "claude_nonzero_exit",
                returncode=proc.returncode,
                stderr=err_msg[:500],
            )

        output = stdout.decode("utf-8", errors="replace").strip() if stdout else None
        return output
