"""Tests for FastWMR randomization bookkeeping.

Planned coverage:
- friction, push, and payload samples are recorded in the expected
  ``env.fastwmr_*`` buffers.
- privileged observations read recorded values rather than resampling them.
- reset and interval randomization update only the intended env indices.
"""

