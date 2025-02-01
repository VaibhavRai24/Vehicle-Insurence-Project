import sys
import logging

def create_error_message(error:Exception, error_detail:sys)->str:
    """
    Extracts detailed error information including file name, line number, and the error message.

    :param error: The exception that occurred.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    line_number = exc_tb.tb_lineno
    error_messages = f"Error has occurred in the python script:[{file_name}] at line number:[{line_number}]: str{error}"
    
    logging.error(error_messages)
    return error_messages

class MyException(Exception):
    def __init__(self, error_message:str, error_detail:sys):
        """
        Initializes the MyException class.

        :param message: The error message.
        :param error_detail: The sys module to access traceback details.
        """
        
        super().__init__(error_message)
        self.error_message = create_error_message(error_message, error_detail)

    def __str__(self) -> str:
        return self.error_message