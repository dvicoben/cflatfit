import logging

handler = logging.StreamHandler()
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(levelname)s:%(name)s: %(message)s"
)
handler.setFormatter(formatter)

def make_logger(name: str, 
                level: int = logging.DEBUG,
                chandler: logging.StreamHandler = handler) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(chandler)
    return logger