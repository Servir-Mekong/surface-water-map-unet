# -*- coding: utf-8 -*-

import os


def file_num_in_folder(_dir):
    return len([name for name in os.listdir(_dir) if os.path.isfile(os.path.join(_dir, name))])
