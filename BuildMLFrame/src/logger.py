import logging

handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger = logging.getLogger("buildml")
logger.addHandler(handler)
logger.setLevel(logging.INFO)