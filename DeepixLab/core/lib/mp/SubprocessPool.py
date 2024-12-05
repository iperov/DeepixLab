from __future__ import annotations

import multiprocessing
from typing import List, Type

from core import ax, mx
from core.lib import os as lib_os


class SubprocessClass(mx.Disposable):
    def __init__(self, ):
        super().__init__()
        self.__process_id : int = ...

    @property
    def process_id(self) -> int: return self.__process_id

    def process(self, job) -> ax.Future:
        """
        Implement your @ax.task job processing here!
        This task can be cancelled.
        """
        raise NotImplementedError()

    @staticmethod
    def _main(scls : Type[SubprocessClass], process_id, prio, h2p_q : multiprocessing.Queue, p2h_q : multiprocessing.Queue, started_ev : multiprocessing.Event, closing_ev : multiprocessing.Event, closed_ev : multiprocessing.Event):
        started_ev.set()
        if prio is not None:
            lib_os.set_process_priority(prio)

        s = scls()
        s.__process_id = process_id

        s._main_task(h2p_q, p2h_q, closing_ev).wait()
        #print('Gracefully quitting subprocess', s.__process_id)
        s.dispose()

        closed_ev.set()

    @ax.task
    def _main_task(self, h2p_q : multiprocessing.Queue, p2h_q : multiprocessing.Queue, closing_ev : multiprocessing.Event):
        job_tasks = {}

        fg = ax.FutureGroup()

        while not closing_ev.is_set():

            while not h2p_q.empty():
                is_cancel, job_id, args, kwargs = h2p_q.get()
                #print("received job", job_id, "in process_id", self.__process_id)

                if is_cancel:
                    if (fut := job_tasks.get(job_id, None)) is not None:
                        #print("Cancelling working job_id", job_id)
                        fut.cancel()
                else:
                    t = job_tasks[job_id] = self.process(*args, **kwargs)
                    t.attach_to(fg)
                    t.call_on_finish(lambda t, job_id=job_id: ( job_tasks.pop(job_id),
                                                                p2h_q.put( (self.__process_id, job_id, t.succeeded, t.result if t.succeeded else t.error) ) ) )

            yield ax.sleep(0)

        fg.dispose()



class SubprocessPool(mx.Disposable):
    """
    Process jobs in subprocesses.

    Jobs cannot be interrupted by cancelling the task.
    """
    Priority = lib_os.ProcessPriority

    class SubprocessDeadException(Exception):
        def __init__(self): super().__init__('Subprocess is dead.')


    def __init__(self,  scls : SubprocessClass,
                        process_frac : float = 1.0,
                        prio : Priority|None = None,
                        graceful_quit : bool = True,
                 ):
        """

            graceful_quit(True)     wait graceful subprocess finalization, otherwise instant kill
        """
        super().__init__()
        self._scls = scls

        self._process_count = process_count = max(1, int(multiprocessing.cpu_count()*process_frac))

        self._prio = prio
        self._graceful_quit = graceful_quit

        self._fg = ax.FutureGroup().dispose_with(self)
        self._bg_thread = ax.Thread().dispose_with(self)

        self._processes : List[multiprocessing.Process|None] = [None] * process_count

        self._evs_started = [None] * process_count
        self._ev_closing = multiprocessing.Event()
        self._evs_closed = [None] * process_count
        self._qs_h2p = [None] * process_count
        self._q_p2h = multiprocessing.Queue()

        self._job_id = 0
        self._n_jobs_in_processes = [0] * process_count

        self._job_result_futures = [ {} for _ in range(process_count)]

        self._bg_task()

    @property
    def process_count(self) -> int: return self._process_count

    def __dispose__(self):
        self._fg.dispose()

        if self._graceful_quit:
            self._ev_closing.set()
            while not all(self._evs_closed[i].is_set() for i, p in enumerate(self._processes) if p is not None and p.is_alive()):
                lib_os.sleep_precise(0.010)

        for i in range(self._process_count):
            self._kill_process(i)

        super().__dispose__()

    def _kill_process(self, process_id : int):
        p = self._processes[process_id]
        if p is not None:
            try:
                p.terminate()
                p.join()
            except:
                pass

            # Cancel pending tasks
            for fut in list(self._job_result_futures[process_id].values()):
                fut.cancel(SubprocessPool.SubprocessDeadException())

            self._processes[process_id] = None
            self._qs_h2p[process_id] = None
            self._evs_started[process_id] = None
            self._evs_closed[process_id] = None

    def _restart_process(self, process_id : int):
        self._kill_process(process_id)

        qs_h2p      = self._qs_h2p[process_id]      = multiprocessing.Queue()
        evs_started = self._evs_started[process_id] = multiprocessing.Event()
        evs_closed  = self._evs_closed[process_id]  = multiprocessing.Event()

        p = multiprocessing.Process(target=SubprocessClass._main, args=(self._scls, process_id, self._prio, qs_h2p,
                                                                        self._q_p2h, evs_started, self._ev_closing,
                                                                        evs_closed), daemon=True)
        p.start()
        self._processes[process_id] = p

    @ax.task
    def _bg_task(self):
        yield ax.attach_to(self._fg)
        yield ax.switch_to(self._bg_thread)

        # Start subprocesses
        for i in range(self._process_count):
            self._restart_process(i)

        job_result_futures = self._job_result_futures
        while True:
            while not self._q_p2h.empty():
                process_id, job_id, job_succeeded, job_result = self._q_p2h.get()
                self._n_jobs_in_processes[process_id] -= 1

                if (fut := job_result_futures[process_id].get(job_id, None)) is not None:
                    if job_succeeded:
                        fut.success(job_result)
                    else:
                        fut.cancel(job_result)

            for i, p in enumerate(self._processes):
                if not p.is_alive():
                    # Process is no longer alive for some reason.
                    print('Subprocess is dead. Restarting.')
                    self._restart_process(i)

            yield ax.sleep(0)

    @ax.task
    def process(self, *args, **kwargs):
        """
        Send a job to process and wait the result.

        Cancelling this task will interrupt the job ASAP.
        """
        yield ax.attach_to(self._fg, detach_parent=False)
        yield ax.switch_to(self._bg_thread)

        try:
            # Wait for all processes to be started
            while not all( (ev := self._evs_started[i]) is not None and ev.is_set() for i in range(self._process_count) ):
                ax.sleep(0)

            # Select started process with the smallest number of tasks
            n_jobs_in_processes = self._n_jobs_in_processes

            process_id = sorted([ (i, n_jobs_in_processes[i]) for i in range(self._process_count)  ], key=lambda x: x[1])[0][0]

            # Generate new job_id
            job_id = self._job_id; self._job_id += 1

            # Send the job to subprocess
            self._qs_h2p[process_id].put( (False, job_id, args, kwargs) )

            # Register in dict
            self._job_result_futures[process_id][job_id] = task = ax.get_current_task()

            # Inc amount of jobs in process
            n_jobs_in_processes[process_id] += 1

            # Wait while this task be finished.
            while True:
                yield ax.sleep(1)

        except ax.TaskFinishException as e:
            # Cancelled by caller or finished by _bg_task
            self._job_result_futures[process_id].pop(job_id)

            if not task.succeeded:
                # Send job_id cancellation signal to subprocess
                self._qs_h2p[process_id].put( (True, job_id, None, None) )
