import os

def check_folder_existing(name):
    """
    Check if the temporary directory exists. If not, create it.

    This function checks if the directory exists in the current working directory.
    If the directory does not exist, it creates it.
    """
    dir_to_check = name
    if not os.path.exists(dir_to_check):
        # If it doesn't exist, create it
        os.makedirs(dir_to_check)
