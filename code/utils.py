import json
import os
import ast
import time

from os.path import splitext, exists




def check_running_file(correct_name='code'):
    """ check that the root directory from where python is running is the right one """

    run_path = os.getcwd()
    to_test = run_path.split("\\")[-1]
    return (to_test == correct_name)

def exist_object(file_name):
    """ check the existence of a .pkl pickle file in the objects folder """

    ext = splitext(file_name)[1]
    if ext != ".pkl":
        raise TypeError("only .pkl files are stored in the objects folder, not {}".format(ext))
    path_to_objects = __file__.split("\\")[:-2]
    path_to_objects.append("objects")
    path_to_objects = "\\".join(path_to_objects)
    return exists(path_to_objects + "\\" + file_name)
