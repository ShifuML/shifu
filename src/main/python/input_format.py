#!/usr/bin/python
# -*- coding: utf-8 -*-
from tensorflow.python.platform import gfile
from StringIO import StringIO
import gzip
import itertools
import datetime
import sys
from abc import ABCMeta, abstractmethod


class InputFormat:
    __metaclass__ = ABCMeta

    @abstractmethod
    def next_batch(self):
        pass

    @abstractmethod
    def initialize(self):
        pass

    @staticmethod
    def get_file_length(file_path):
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            return sum(1 for line in gf)

    @staticmethod
    def get_split(file_path, start, end):
        lines = []
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            for line in itertools.islice(gf, start, end):
                lines.append(line)
        return lines

    @staticmethod
    def get_first_line(file_path):
        with gfile.Open(file_path, 'rb') as f:
            gf = gzip.GzipFile(fileobj=StringIO(f.read()))
            line = gf.readline()
            return line

    @staticmethod
    def tprint(content, log_level="INFO"):
        sys_time = datetime.datetime.now()
        print(str(sys_time) + " " + log_level + " " + " [Shifu.Tensorflow.train] " + str(content))
        sys.stdout.flush()


if __name__ == '__main__':
    print InputFormat.get_first_line("hdfs://horton/user/website/ModelSets/demo/tmp/NormalizedData/part-m-00000.gz")
