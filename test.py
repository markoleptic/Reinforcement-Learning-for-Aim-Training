from FileModHandler import FileModified

def file_modified():
    print("File Modified!")
    return False

fileModifiedHandler = FileModified(r"test file.txt", file_modified)
fileModifiedHandler.start()