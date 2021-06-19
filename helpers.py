import logging
import sys


class outLogger:
    def __init__(self):
        self.l = self.create_logger()

    def create_logger(self):
        a_logger = logging.getLogger()
        a_logger.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        a_logger.addHandler(stdout_handler)
        return a_logger

    def __call__(self, x):
        return self.l.debug("LOG: " + str(x))
