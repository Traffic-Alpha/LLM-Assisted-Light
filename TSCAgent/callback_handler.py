'''
@Author: WANG Maonan
@Date: 2023-09-04 20:51:07
@Description: callback function
@LastEditTime: 2023-09-18 21:41:51
'''
from loguru import logger
from langchain.callbacks import FileCallbackHandler

def create_file_callback(logfile:str="output.log"):
    logger.add(logfile, colorize=False, enqueue=True)
    handler = FileCallbackHandler(logfile)
    return handler