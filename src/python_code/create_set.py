#!/usr/bin/env python3
from DataBase_Manager.file_manager import FileManager
import sys


path = sys.argv[1]
fm = FileManager(path)
print(fm.create_set())

