"""Ordinary transition replay used by FastSAC and FastWMR.

The baseline initially stores ``(o, a, r, o_next, terminated, truncated)``.
Integrated FastWMR stores detached ``x``/``x_next`` in the corresponding state
slots, optionally with an estimator-version tag to monitor feature staleness.
Recurrent hidden/cell tensors never belong in this buffer.
"""
