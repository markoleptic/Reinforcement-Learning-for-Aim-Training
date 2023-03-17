from FileModHandler import FileModified

def file_modified():
    print("File Modified!")
    return False

fileModifiedHandler = FileModified(r"U:/UnrealGameBuilds/Debug/Windows/BeatShot/AccuracyMatrix.csv", file_modified)
fileModifiedHandler.start()