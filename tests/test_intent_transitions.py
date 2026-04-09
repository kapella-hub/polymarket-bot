"""Tests for intent state machine transition enforcement."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from src.execution.intent import IntentManager, _TRANSITIONS
from src.db.models import IntentState, InvalidationReason


class TestTransitionEnforcement:
    @pytest.mark.asyncio
    async def test_invalid_transition_rejected(self):
        """CREATED -> EXECUTED should be rejected (must go through ARMED)."""
        mgr = IntentManager()

        with patch("src.execution.intent.async_session") as mock_sess:
            mock_session = AsyncMock()
            mock_sess.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_state = AsyncMock(return_value=IntentState.CREATED)
            mock_repo.update_state = AsyncMock()

            with patch("src.execution.intent.IntentRepository", return_value=mock_repo):
                result = await mgr.transition(1, IntentState.EXECUTED)

        assert result is False

    @pytest.mark.asyncio
    async def test_valid_transition_accepted(self):
        """CREATED -> ARMED should be accepted."""
        mgr = IntentManager()

        with patch("src.execution.intent.async_session") as mock_sess:
            mock_session = AsyncMock()
            mock_sess.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_state = AsyncMock(return_value=IntentState.CREATED)
            mock_repo.update_state = AsyncMock()

            with patch("src.execution.intent.IntentRepository", return_value=mock_repo):
                result = await mgr.transition(1, IntentState.ARMED)

        assert result is True

    @pytest.mark.asyncio
    async def test_terminal_state_rejects_all(self):
        """EXECUTED is terminal — should reject any further transition."""
        mgr = IntentManager()

        with patch("src.execution.intent.async_session") as mock_sess:
            mock_session = AsyncMock()
            mock_sess.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_state = AsyncMock(return_value=IntentState.EXECUTED)
            mock_repo.update_state = AsyncMock()

            with patch("src.execution.intent.IntentRepository", return_value=mock_repo):
                result = await mgr.transition(1, IntentState.CANCELLED)

        assert result is False

    @pytest.mark.asyncio
    async def test_nonexistent_intent_returns_false(self):
        """Missing intent should return False, not crash."""
        mgr = IntentManager()

        with patch("src.execution.intent.async_session") as mock_sess:
            mock_session = AsyncMock()
            mock_sess.return_value.__aenter__ = AsyncMock(return_value=mock_session)
            mock_sess.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_repo = MagicMock()
            mock_repo.get_state = AsyncMock(return_value=None)

            with patch("src.execution.intent.IntentRepository", return_value=mock_repo):
                result = await mgr.transition(999, IntentState.ARMED)

        assert result is False
