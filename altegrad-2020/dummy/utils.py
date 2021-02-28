import json
import os
import ast
import time

from os import path
from os.path import splitext, exists



def count_lines(file=None, file_name=None):
    if file:
        pass
    elif file_name:
        file = open(file_name, "r", encoding='utf8')
    else:
        raise TypeError("count_lines missing at least one argument, should be a file or a string")
    line_count = 0
    for line in file:
        if line != "\n":
            line_count += 1
    if file_name:
        file.close()

    return line_count


def check_running_file(correct_name='altegrad-2020'):
    run_path = os.getcwd()
    to_test = run_path.split("\\")[-1]
    return (to_test == correct_name)


def how_many_word_embedded():
    with open("word_embeddings.json", 'r') as f:
        embeddings = json.load(f)
    print(len(list(embeddings.keys())))


def exist_object(file_name):
    """ Check the existence of a .pkl pickle file in the objects folder """
    ext = splitext(file_name)[1]
    if ext != ".pkl":
        raise TypeError("only .pkl files are stored in the objects folder, not {}".format(ext))
    path_to_objects = __file__.split("\\")[:-2]
    path_to_objects.append("objects")
    path_to_objects = "\\".join(path_to_objects)
    return exists(path_to_objects + "\\" + file_name)


def chrono():
    string_list = "['word1', 'word2', 'word3', 'word4', 'word5', 'word6', 'word7', 'word8', 'word9', 'word10', 'word11']"
    my_list = ast.literal_eval(string_list)
    string_string = ",".join(my_list)
    t1 = time.time()
    for _ in range(100000):
        ast.literal_eval(string_list)
    t2 = time.time()
    delta = t2 - t1
    print("time token by literal_eval :", delta)

    t1 = time.time()
    for _ in range(100000):
        string_list.split(",")
    t2 = time.time()
    delta = t2 - t1
    print("time token by split :", delta)


def print_lines(file_name, num_lines=5):
    with open(file_name, "r", encoding="utf8") as f:
        reader = f.readlines()
        for i in range(num_lines):
            print(reader[i])

