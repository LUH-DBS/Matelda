"""Setup logging configurations """
import logging
import logging.handlers
import os


def setup_logging(logs_dir: str):
    """
    Setup logging configurations

    Args:
        logs_dir (str): Path to the directory where the logs will be stored
    """
    # Logging Configurations
    os.makedirs(logs_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.environ.get("LOGFILE", os.path.join(logs_dir, "app.log")),
        level=logging.DEBUG,
        format="%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s",
    )
