#!/usr/bin/env python
# coding:utf-8

import logging

logging_level = {'debug': logging.DEBUG,
                 'info': logging.INFO,
                 'warning': logging.WARNING,
                 'error': logging.ERROR,
                 'critical': logging.CRITICAL}


def debug(msg):
    logging.debug(msg)
    print('DEBUG: ', msg)


def info(msg):
    logging.info(msg)
    print('INFO: ', msg)


def warning(msg):
    logging.warning(msg)
    print('WARNING: ', msg)


def error(msg):
    logging.error(msg)
    print('ERROR: ', msg)


def fatal(msg):
    logging.critical(msg)
    print('FATAL: ', msg)


class Logger(object):
    def __init__(self, config):
        """
        set the logging module
        :param config: helper.configure, Configure object
        """
        super(Logger, self).__init__()
        assert config.log.level in logging_level.keys()
        logging.getLogger('').handlers = []
        logging.basicConfig(filename=config.log.filename,
                            level=logging_level[config.log.level],
                            format='%(asctime)s - %(levelname)s : %(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S')

