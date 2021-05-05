# istarmap.py for Python 3.8+
import multiprocessing.pool as mpp


def multiprocessing_istarmap(self, func, iterable, chunksize=1):
    """
    starmap-version of imap
    This is only for possibility of displaying progress bar in jupyter notebook during
    multiprocessing of images to superpixels graphs.
    Source: https://stackoverflow.com/questions/57354700/starmap-combined-with-tqdm/57364423#57364423
    """
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


# to apply patch:
# mpp.Pool.istarmap = better_istarmap
