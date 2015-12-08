#LOGGING BOILERPLATE
from datetime import datetime
start_time = datetime.now()
import os

#logging boilerplate.  To log anything type log.info('stuff you want to log')
import logging as log


def create_log(file_name):
	#make log file if it doesn't exist
	if not os.path.exists('../logs'):
		os.makedirs('../logs')

	#log file with filename, HMS time
	log.basicConfig(filename='logs/%s%s.log' %(file_name.split('.')[0],start_time.strftime('_%Y%m%d_%H%M%S')),level=log.DEBUG)

	return log

def plog(msg):
	print msg
	log.info(msg)