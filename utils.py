from time import time


class Chrono:
    def __init__(self):
        self.start_time = time()

    def get_elapsed_time(self):
        return time() - self.start_time
