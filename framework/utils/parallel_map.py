from torch import multiprocessing
from typing import Iterable, Callable, Any, List, Optional
import time
import math


class ParallelMapPool:
    def __init__(self, callback: Callable[[Any], None], max_parallel: Optional[int] = None, debug: bool = False):
        self.n_proc = max_parallel or multiprocessing.cpu_count()
        self.callback = callback
        self.debug = debug

    def __enter__(self):
        if self.debug:
            return self

        self.in_queues = [multiprocessing.Queue() for _ in range(self.n_proc)]
        self.out_queues = [multiprocessing.Queue() for _ in range(self.n_proc)]

        def run_process(in_q: multiprocessing.Queue, out_q: multiprocessing.Queue):
            while True:
                args = in_q.get()
                if args is None:
                    break
                res = [self.callback(a) for a in args]
                out_q.put(res)

        self.processes = [multiprocessing.Process(target=run_process, args=(iq, oq), daemon=True)
                          for iq, oq in zip(self.in_queues, self.out_queues)]
        for p in self.processes:
            p.start()

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.debug:
            return

        for q in self.in_queues:
            q.put(None)

        for p in self.processes:
            p.join()

        self.in_queues = None
        self.out_queues = None

    def map(self, args: List) -> List:
        if self.debug:
            return [self.callback(a) for a in args]

        a_per_proc = math.ceil(len(args) / self.n_proc)
        chunks = [args[a : a + a_per_proc] for a in range(0, len(args), a_per_proc)]

        for q, c in zip(self.in_queues, chunks):
            q.put(c)

        res = [q.get() for q in self.out_queues]
        return sum(res, [])


def parallel_map(tasks: Iterable, callback = Callable[[Any], None], max_parallel: int = 32) -> List:
    limit = min(multiprocessing.cpu_count(), max_parallel)
    processes: List[multiprocessing.Process] = []
    queues: List[multiprocessing.Queue] = []
    indices: List[int] = []
    tlist = [t for t in tasks]
    res = [None] * len(tlist)
    curr = 0

    def process_return(q, arg):
        res = callback(arg)
        q.put(res)

    while curr < len(tlist):
        if len(processes) == limit:
            ended = []
            for i, q in enumerate(queues):
                if not q.empty():
                    processes[i].join()
                    ended.append(i)
                    res[indices[i]] = q.get()

            for i in sorted(ended, reverse=True):
                processes.pop(i)
                queues.pop(i)
                indices.pop(i)

            if not ended:
                time.sleep(0.1)
                continue

        queues.append(multiprocessing.Queue())
        indices.append(curr)
        processes.append(multiprocessing.Process(target=process_return, args=(queues[-1], tlist[curr]), daemon=True))
        processes[-1].start()

        curr += 1

    for i, p in enumerate(processes):
        res[indices[i]] = queues[i].get()
        p.join()

    return res