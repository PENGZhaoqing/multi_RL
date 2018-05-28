from __future__ import absolute_import, division, print_function
import os
import inspect
import numpy as np
import time

import math


def print_params(logging, network):
    network_params = network.get_params()
    assert network_params is not None, "Fatal Error!"
    logging.info("Nework Params: ")
    for k, v in network_params[0].items():
        logging.info("   %s: %s" % (k, v))

    optimizer = network._optimizer
    logging.info("Optimizer Params: " + str(optimizer.__class__))
    for k, v in optimizer.kwargs.items():
        logging.info("   %s: %s" % (k, v))

    logging.info("   wd: " + str(optimizer.wd))
    logging.info("   lr: " + str(optimizer.lr))


def print_params1(logging, network):
    arg_params = network.arg_dict
    aux_params = network.aux_dict
    assert arg_params is not None, "Fatal Error!"
    logging.info("Nework Params: ")
    for k, v in arg_params.items():
        logging.info("   %s: %s" % (k, v))

    logging.info("Nework Aux Params: ")
    for k, v in aux_params.items():
        logging.info("   %s: %s" % (k, v))

    optimizer = network.updater.optimizer
    logging.info("Optimizer Params: " + str(optimizer.__class__))
    for k, v in optimizer.kwargs.items():
        logging.info("   %s: %s" % (k, v))

    logging.info("   wd: " + str(optimizer.wd))
    logging.info("   lr: " + str(optimizer.lr))


def logging_config(logging, config):
    dir, folder_name, name = config.dir, config.save_log_path, config.save_log

    level = logging.DEBUG
    if name is None or folder_name is None:
        name = inspect.stack()[1][1].split('.')[0]
        folder_name = inspect.stack()[1][1].split('.')[0]
    # folder = os.path.join(os.getcwd(), folder_name)
    # print(os.getcwd())
    folder = folder_name
    # print(folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # print(os.path.exists(folder))
    now = time.strftime("%m:%d:%H:%M", time.localtime(time.time()))
    logpath = os.path.join(folder, name + "_" + str(now) + ".log")
    print("All Logs will be saved to %s" % logpath)
    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)
    return folder


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


def count_distant(agent1, agent2):
    try:
        dis = math.sqrt((agent1.pos[0] - agent2.pos[0]) ** 2 + (agent1.pos[1] - agent2.pos[1]) ** 2)
    except (AttributeError, TypeError):
        raise AssertionError('Object should extend from Agent Class')
    return dis


def normalization(vector):
    norm = math.sqrt(vector[0] ** 2 + vector[1] ** 2)
    vector[0] /= norm
    vector[1] /= norm
    return vector


def _2d_list(n):
    return [[] for _ in range(n)]
