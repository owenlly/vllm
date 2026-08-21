# SPDX-License-Identifier: Apache-2.0
from typing import NamedTuple


class ExternalLoadCompletionNotice(NamedTuple):
    """One rank-local successful external LOAD completion hint."""

    epoch: int
    rank: int
    job_id: int
    req_id: str
    completed_at_s: float
