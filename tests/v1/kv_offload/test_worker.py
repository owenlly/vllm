# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.v1.kv_offload.base import LoadStoreSpec
from vllm.v1.kv_offload.worker.worker import (
    OffloadingHandler,
    OffloadingWorker,
    TransferResult,
    TransferSpec,
)


class LoadStoreSpec1(LoadStoreSpec):
    def __init__(
        self,
        submit_success: bool = True,
        async_success: bool = True,
        exception: bool = False,
    ):
        self.finished = False
        self.submit_success = submit_success
        self.async_success = async_success
        self.exception = exception

    @staticmethod
    def medium() -> str:
        return "1"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class LoadStoreSpec2(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "2"

    def __repr__(self):
        return f"{self.medium()}: {id(self)}"


class GPUStoreSpec(LoadStoreSpec):
    @staticmethod
    def medium() -> str:
        return "GPU"


class GPUStoreHandler(OffloadingHandler):
    def __init__(self):
        self.jobs: set[int] = set()
        self.finished: list[TransferResult] = []
        self.whole_waits: list[set[int]] = []
        self.retired: list[set[int]] = []
        self.fail_wait = False

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        self.jobs.add(job_id)
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = self.finished
        self.finished = []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        if self.fail_wait:
            raise RuntimeError(f"cannot prove source safety for {job_ids}")
        self.whole_waits.append(set(job_ids))

    def retire_store_source_jobs(self, job_ids: set[int]) -> None:
        self.retired.append(set(job_ids))


class OffloadingHandler1To2(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec1)
        assert isinstance(dst, LoadStoreSpec2)

        if src.exception:
            raise Exception("An expected exception. Don't worry!")
        if not src.submit_success:
            return False

        self.transfers[job_id] = src
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            spec = self.transfers.get(job_id)
            if spec:
                assert spec.finished


class OffloadingHandler2To1(OffloadingHandler):
    def __init__(self):
        self.transfers: dict[int, LoadStoreSpec1] = {}

    def transfer_async(self, job_id: int, spec: TransferSpec) -> bool:
        src, dst = spec
        assert isinstance(src, LoadStoreSpec2)
        assert isinstance(dst, LoadStoreSpec1)

        self.transfers[job_id] = dst
        return True

    def get_finished(self) -> list[TransferResult]:
        finished = []
        for job_id, spec in list(self.transfers.items()):
            if spec.finished:
                finished.append((job_id, spec.async_success))
                del self.transfers[job_id]
        return finished

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            spec = self.transfers.get(job_id)
            if spec:
                assert spec.finished


def test_offloading_worker():
    """
    Tests OffloadingWorker with 2 handlers.
    One handler performs 1->2 transfers, and the other handles 2->1.
    """
    worker = OffloadingWorker()
    handler1to2 = OffloadingHandler1To2()
    handler2to1 = OffloadingHandler2To1()
    worker.register_handler(LoadStoreSpec1, LoadStoreSpec2, handler1to2)
    worker.register_handler(LoadStoreSpec2, LoadStoreSpec1, handler2to1)

    # 1st transfer 1->2 (exception)
    src1 = LoadStoreSpec1(exception=True)
    dst1 = LoadStoreSpec2()
    assert not worker.transfer_async(1, (src1, dst1))

    # 2ed transfer 1->2 (failure to submit)
    src2 = LoadStoreSpec1(submit_success=False)
    dst2 = LoadStoreSpec2()
    assert not worker.transfer_async(2, (src2, dst2))

    # 3rd transfer 1->2 (failure)
    src3 = LoadStoreSpec1(async_success=False)
    dst3 = LoadStoreSpec2()
    assert worker.transfer_async(3, (src3, dst3))

    # 4th transfer 1->2 (success)
    src4 = LoadStoreSpec1()
    dst4 = LoadStoreSpec2()
    worker.transfer_async(4, (src4, dst4))
    assert set(handler1to2.transfers.keys()) == {3, 4}

    # 5th transfer 2->1
    src5 = LoadStoreSpec2()
    dst5 = LoadStoreSpec1()
    worker.transfer_async(5, (src5, dst5))
    assert set(handler2to1.transfers.keys()) == {5}

    # no transfer completed yet
    assert worker.get_finished() == []

    # complete 3rd, 4th
    src3.finished = True
    src4.finished = True

    # 6th transfer 1->2
    src6 = LoadStoreSpec1()
    dst6 = LoadStoreSpec2()
    worker.transfer_async(6, (src6, dst6))

    # 7th transfer 2->1
    src7 = LoadStoreSpec2()
    dst7 = LoadStoreSpec1()
    worker.transfer_async(7, (src7, dst7))

    # 6th and 7th transfers started
    assert 6 in handler1to2.transfers
    assert 7 in handler2to1.transfers

    # verify result of 3rd and 4th transfers
    assert sorted(worker.get_finished()) == [(3, False), (4, True)]

    # complete 6th and 7th transfers
    src6.finished = True
    dst7.finished = True
    assert sorted(worker.get_finished()) == [(6, True), (7, True)]


def test_store_source_wait_defaults_to_whole_job_and_retires_mapping():
    worker = OffloadingWorker()
    handler = GPUStoreHandler()
    worker.register_handler(GPUStoreSpec, LoadStoreSpec2, handler)
    assert worker.transfer_async(11, (GPUStoreSpec(), LoadStoreSpec2()))
    terminal = TransferResult(job_id=11, success=True)
    handler.finished.append(terminal)
    assert worker.get_finished() == [terminal]
    assert 11 in worker._store_job_to_handler

    worker.wait_store_source_safe({11: (3, 4)})
    assert handler.whole_waits == [{11}]

    worker.retire_store_source_jobs({11})
    assert handler.retired == [{11}]
    assert worker._store_job_to_handler == {}
    with pytest.raises(RuntimeError, match="unknown job 11"):
        worker.wait_store_source_safe({11: (3,)})


def test_store_source_wait_rejects_missing_job():
    worker = OffloadingWorker()
    handler = GPUStoreHandler()
    worker.register_handler(GPUStoreSpec, LoadStoreSpec2, handler)

    with pytest.raises(RuntimeError, match="unknown job 12"):
        worker.wait_store_source_safe({12: (7,)})


def test_store_source_default_fallback_failure_is_fail_stop():
    worker = OffloadingWorker()
    handler = GPUStoreHandler()
    worker.register_handler(GPUStoreSpec, LoadStoreSpec2, handler)
    assert worker.transfer_async(13, (GPUStoreSpec(), LoadStoreSpec2()))

    handler.fail_wait = True
    with pytest.raises(RuntimeError, match="cannot prove source safety"):
        worker.wait_store_source_safe({13: (8,)})
    assert 13 in worker._store_job_to_handler


def test_duplicate_store_job_id_is_rejected_and_retire_is_idempotent():
    worker = OffloadingWorker()
    handler = GPUStoreHandler()
    worker.register_handler(GPUStoreSpec, LoadStoreSpec2, handler)
    spec = (GPUStoreSpec(), LoadStoreSpec2())
    assert worker.transfer_async(14, spec)
    with pytest.raises(ValueError, match="duplicate STORE transfer job ID 14"):
        worker.transfer_async(14, spec)

    worker.retire_store_source_jobs({14})
    worker.retire_store_source_jobs({14})
    assert handler.retired == [{14}]
