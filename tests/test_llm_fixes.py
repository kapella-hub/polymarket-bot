"""Tests for LLM pipeline bug fixes."""

import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from src.llm.claude_runner import ClaudeRunner


class TestNonZeroExitHandling:
    @pytest.mark.asyncio
    async def test_nonzero_exit_returns_none(self):
        """CLI non-zero exit should return None, not partial stdout."""
        runner = ClaudeRunner(timeout=10, max_retries=0)

        mock_proc = MagicMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(
            return_value=(b'{"probability": 0.99}', b"Error: rate limited")
        )

        with patch("src.llm.claude_runner.asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await runner._run_cli("test prompt")

        assert result is None


class TestTemplateNoneGuard:
    def test_generic_template_handles_none_price(self):
        """Template should not crash when yes_price is None."""
        from jinja2 import Environment, FileSystemLoader
        import pathlib

        template_dir = pathlib.Path(__file__).parent.parent / "prompts"
        if not template_dir.exists():
            pytest.skip("prompts directory not found")

        env = Environment(loader=FileSystemLoader(str(template_dir)))
        template = env.get_template("generic.j2")

        # This should NOT raise TypeError
        result = template.render(
            system_prompt="test",
            question="Will X happen?",
            yes_price=None,
            no_price=None,
            category="Test",
            end_date=None,
            time_remaining=None,
            description=None,
            resolution_source=None,
            volume=None,
            liquidity=None,
            enrichment_context=None,
        )
        assert "Will X happen?" in result
        assert "unavailable" in result
