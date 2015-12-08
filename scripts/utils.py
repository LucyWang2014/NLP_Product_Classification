#LOGGING BOILERPLATE
from datetime import datetime
import os
import logging as log

#logging boilerplate.  To log anything type log.info('stuff you want to log')

def create_log(script_name):
    start_time = datetime.now()
    #make log file if it doesn't exist
    if not os.path.exists('../logs'):
        os.makedirs('../logs')

    #log file with filename, HMS time
    log_time = start_time.strftime('_%Y%m%d_%H%M%S')
    log_fname = '../logs/%s%s.log' %(script_name.split('.')[0],log_time)
    log.basicConfig(filename=log_fname,level=log.DEBUG)


def plog(msg):
    print msg
    log.info(msg)

def fplog(msg):
    print(msg)
    log.info(msg)