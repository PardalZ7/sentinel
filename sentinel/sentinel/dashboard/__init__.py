"""Sentinel Dashboard — lightweight observation layer.

The dashboard is completely passive with respect to agent/cortex processing:
all state is read from shared in-memory structures written by the agents,
never from the processing hot path.
"""
