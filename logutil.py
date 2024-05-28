import logging
import logging.handlers
import os
import time
 
class logs(object):
    def __init__(self):
        self.logger = logging.getLogger("")
        # Set the level of output
        LEVELS = {'NOSET': logging.NOTSET,
                  'DEBUG': logging.DEBUG,
                  'INFO': logging.INFO,
                  'WARNING': logging.WARNING,
                  'ERROR': logging.ERROR,
                  'CRITICAL': logging.CRITICAL}
        # Create file directory
        logs_dir="logs"
        if os.path.exists(logs_dir) and os.path.isdir(logs_dir):
            pass
        else:
            os.mkdir(logs_dir)
        # Modify the log save location
        timestamp=time.strftime("%Y-%m-%d",time.localtime())
        logfilename='%s.txt' % timestamp
        logfilepath=os.path.join(logs_dir,logfilename)
        rotatingFileHandler = logging.handlers.RotatingFileHandler(filename =logfilepath,
                                                                   encoding='utf-8',
                                                                   maxBytes = 1024 * 1024 * 50,
                                                                   backupCount = 5)
        # Set output format
        formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
        rotatingFileHandler.setFormatter(formatter)

        console = logging.StreamHandler()
        console.setLevel(logging.NOTSET)
        console.setFormatter(formatter)

        self.logger.addHandler(rotatingFileHandler)
        self.logger.addHandler(console)
        self.logger.setLevel(logging.NOTSET)
 
    def info(self, *message):
        msg = ''
        for m in message:
            msg = msg+' '+str(m)  # int
        self.logger.info(msg)
 
    def debug(self, message):
        self.logger.debug(message)
 
    def warning(self, message):
        self.logger.warning(message)
 
    def error(self, message):
        self.logger.error(message)