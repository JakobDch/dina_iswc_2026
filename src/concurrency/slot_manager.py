"""Manages slot allocation for concurrent experiment runs.

This module provides a SlotManager that allocates exclusive container slots
to concurrent experiment runs, ensuring that Mapping Optimizer modifications
don't affect other running queries.

Usage:
    from src.concurrency.slot_manager import get_slot_manager
    from src.config import slot_context

    slot_manager = get_slot_manager(num_slots=5)
    await slot_manager.initialize()

    async with slot_manager.acquire_slot() as slot_id:
        with slot_context(slot_id):
            # This run has exclusive access to slot_id's containers
            result = await run_experiment(...)
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

logger = logging.getLogger(__name__)


class SlotManager:
    """Manages a pool of container slots for concurrent execution.

    Each slot represents an exclusive set of OnTop containers that
    can be used by one concurrent run at a time. When a run acquires
    a slot, it has exclusive access to that slot's containers until
    the slot is released.

    This prevents the Mapping Optimizer from affecting other concurrent
    queries when it modifies mappings and restarts containers.
    """

    def __init__(self, num_slots: int = 5):
        """Initialize the slot manager.

        Args:
            num_slots: Number of slots available (should match docker-compose.slots.yml)
        """
        self.num_slots = num_slots
        self._available_slots: asyncio.Queue[int] = asyncio.Queue()
        self._initialized = False
        self._lock = asyncio.Lock()

    async def initialize(self) -> None:
        """Initialize the slot pool with available slots.

        Must be called before acquiring slots. Safe to call multiple times.
        """
        async with self._lock:
            if self._initialized:
                return

            for i in range(self.num_slots):
                await self._available_slots.put(i)

            self._initialized = True
            logger.info(f"SlotManager initialized with {self.num_slots} slots")

    @asynccontextmanager
    async def acquire_slot(self) -> AsyncIterator[int]:
        """Acquire an exclusive slot for the duration of a run.

        This is an async context manager that blocks until a slot is available,
        then yields the slot ID. The slot is automatically released when the
        context exits (even on exception).

        Usage:
            async with slot_manager.acquire_slot() as slot_id:
                # Use slot_id for this run
                ...

        Yields:
            The slot ID (0 to num_slots-1)

        Raises:
            RuntimeError: If the slot manager hasn't been initialized
        """
        if not self._initialized:
            raise RuntimeError("SlotManager not initialized. Call initialize() first.")

        slot_id = await self._available_slots.get()
        logger.debug(f"Acquired slot {slot_id} ({self._available_slots.qsize()} remaining)")

        try:
            yield slot_id
        finally:
            await self._available_slots.put(slot_id)
            logger.debug(f"Released slot {slot_id} ({self._available_slots.qsize()} available)")

    def get_available_count(self) -> int:
        """Get the number of currently available slots.

        Returns:
            Number of slots not currently in use
        """
        return self._available_slots.qsize()

    @property
    def is_initialized(self) -> bool:
        """Check if the slot manager has been initialized."""
        return self._initialized


# Global slot manager instance
_slot_manager: SlotManager | None = None


def get_slot_manager(num_slots: int = 5) -> SlotManager:
    """Get or create the global slot manager.

    The slot manager is created once and reused. The num_slots parameter
    is only used on first creation.

    Args:
        num_slots: Number of slots (only used on first call)

    Returns:
        The global SlotManager instance
    """
    global _slot_manager
    if _slot_manager is None:
        _slot_manager = SlotManager(num_slots)
    return _slot_manager


def reset_slot_manager() -> None:
    """Reset the global slot manager (for testing)."""
    global _slot_manager
    _slot_manager = None
