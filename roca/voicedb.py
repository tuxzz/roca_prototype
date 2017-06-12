import os, pickle
from .common import *

class DevelopmentVoiceDB:
    def __init__(self, path):
        self.path = path
        if(not os.path.exists(path)):
            os.makedirs(path)
        if(not os.path.isdir(path)):
            raise IOError("Not a sparse voice database directory.")

    def getFullPath(self, path):
        return os.path.normpath(os.path.join(self.path, path))

    def saveObject(self, path, obj):
        path = self.getFullPath(path)
        with open(path, "wb") as f:
            pickle.dump(obj, f)

    def loadObject(self, path):
        path = self.getFullPath(path)
        with open(path, "rb") as f:
            return pickle.load(f)

    def hasObject(self, path):
        path = self.getFullPath(path)
        return os.path.isfile(path)

    def objectList(self):
        out = []
        for root, dirList, fileList in os.walk(self.path):
            l = []
            for fileName in fileList:
                l.append(fileName)
            out.append((os.path.relpath(root, self.path), l))
        return out
