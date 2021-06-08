#!/usr/bin/env python3
from DataBase_Manager.beta_manager import gather_data
import sys

experiment_path = sys.argv[1]
csv_path = sys.argv[2]

gather_data(experiment_path, save_file=csv_path)