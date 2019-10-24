  
import sys
import os
from datetime import datetime
import logging


class Logger(object):

    def __init__(self):
        self.logger = logging.getLogger()        
        self.logger.setLevel(logging.INFO)

    def setup(self, dirname, name):

        os.makedirs(dirname, exist_ok=True)
        path = f'{dirname}/{name}.log'

        # log formater
        format='%(asctime)s- %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        formatter = logging.Formatter(format)

        # write to file
        file_handler = logging.FileHandler(path, 'a')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # stream to terminal
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)


        log.info('')
        log.info('--- %s ---' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        log.info(''.join(sys.argv))
        log.info('logpath: %s' % path)

logger = Logger()
log = logger.logger