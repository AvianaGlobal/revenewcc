import lo

log_file = 'log.txt'
logging.basicConfig(filename=log_file, level=logging.INFO)
handler = logging.StreamHandler()
logger = logging.getLogger()
logger.addHandler(handler)