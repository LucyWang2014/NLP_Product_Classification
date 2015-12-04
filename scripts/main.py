'''
main.py
End-to-end script for running all processes.  
'''
__author__='Charlie Guthrie'

from datetime import datetime
start_time = datetime.now()

import os
#make log file if it doesn't exist
if not os.path.exists('logs'):
    os.makedirs('logs')
#logging boilerplate.  To log anything type log.info('stuff you want to log')
import logging as log
#log file with filename, HMS time
log.basicConfig(filename='logs/%s%s.log' %(__file__.split('.')[0],start_time.strftime('%H%M%S')),level=log.DEBUG)


import data_prep
import models

data = data_prep.main()

models(data,options)