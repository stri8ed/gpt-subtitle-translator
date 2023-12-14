import logging

logger = logging.getLogger('gpt-translator')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter('%(levelname)s - %(message)s')
handler.setFormatter(formatter)

logger.addHandler(handler)