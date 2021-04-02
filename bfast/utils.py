import logging

class Logger():
    def __init__(self, verbosity, name):
        self.verbosity = verbosity
        if self.verbosity == 2:
            log_level = "DEBUG"
        elif self.verbosity == 1:
            log_level = "INFO"
        elif self.verbosity == 0:
            log_level = "WARNING"
        else:
            raise ValueError("Invalid verbosity level: {}. Must be between 0 and 2")
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(name)

    def warning(self, msg):
        if self.verbosity >= 0:
            self.logger.warning(msg)

    def info(self, msg):
        if self.verbosity > 0:
            self.logger.info(msg)

    def debug(self, msg):
        if self.verbosity > 1:
            self.logger.debug(msg)
