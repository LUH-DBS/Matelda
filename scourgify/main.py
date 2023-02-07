from configparser import ConfigParser
import app_logger
from scourgify.load_module import read_data

configs = ConfigParser()
configs.read("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/EDS/config.ini")

logs_dir = configs["DIRECTORIES"]["logs_dir"]
logger = app_logger.get_logger(logs_dir)

df = read_data.read_csv("/Users/fatemehahmadi/Documents/Github-Private/ED-Scale/EDS/raha-datasets/parent/beers/beers.csv", False)
print(df.shape)