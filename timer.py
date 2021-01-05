import time


class Timer:

    start = None
    offset = 0

    pauseTime = 0

    def __init__(self):
        self.start = time.time()

    def elapsed(self):
        return time.time() - self.start - self.offset

    def pause(self):
        self.pauseTime = time.time()

    def resume(self):
        self.offset += time.time() - self.pauseTime
