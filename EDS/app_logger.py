import logging
import logging.handlers
import os


def get_logger(logs_dir):
    # Logging Configurations
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    handler = logging.handlers.WatchedFileHandler(os.environ.get("LOGFILE", os.path.join(logs_dir, "app.log")))
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s"))
    app_logger = logging.getLogger()
    app_logger.setLevel(os.environ.get("LOGLEVEL", "INFO"))
    app_logger.addHandler(handler)

    return app_logger
