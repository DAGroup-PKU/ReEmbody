"""Cross-process request slots shared with existing frozen campaigns."""
import contextlib
import fcntl
import pathlib
import time


@contextlib.contextmanager
def api_slot(config):
    root = pathlib.Path(config.get('api_budget_root') or config.get('batch_root') or config['run_root']) / 'api_slots'
    root.mkdir(parents=True, exist_ok=True)
    limit = config.get('k3_concurrency', 2)
    if type(limit) is not int or limit < 1:
        raise ValueError('API concurrency must be a positive integer')
    started = time.monotonic()
    acquired = None
    slot = None
    while acquired is None:
        for index in range(limit):
            handle = (root / str(index)).open('a')
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
            except BaseException:
                handle.close()
                raise
            else:
                acquired, slot = handle, index
                break
        if acquired is None:
            time.sleep(.15)
    try:
        yield {'slot': slot, 'queue_wait_s': time.monotonic() - started,
               'concurrency': limit}
    finally:
        acquired.close()
