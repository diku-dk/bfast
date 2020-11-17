import logging
import argparse

def logging_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log",
                        default="warning",
                        help="set the logging level, default is WARNING")
    args = parser.parse_args()
    log_level = args.log
    # format_string = "%(levelname)s: %(message)s"
    # logging.basicConfig(format=format_string, level=getattr(logging, log_level.upper()))
    logging.basicConfig(level=getattr(logging, log_level.upper()))
