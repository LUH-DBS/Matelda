import logging
import logging.handlers
import os


def get_logger():
    # Logging Configurations
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", "logs/app.log"))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"))
    app_logger = logging.getLogger()
    app_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    app_logger.addHandler(handler)

    return app_logger
