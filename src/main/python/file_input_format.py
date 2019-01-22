from input_format import InputFormat
from tensorflow.python.platform import gfile
import os
import sys
import math
import random


class FileInputFormat(InputFormat):
    def __init__(self, context):
        self._context = context
        self._root_folder = context["data_root_folder"]
        self._batch_num = context["batch_size"]
        self._delimiter = context["delimiter"]
        self._feature_column_nums = context["feature_column_nums"]
        self._sample_weight_column_num = context["sample_weight_column_num"]
        self._target_index = context["target_index"]
        self._valid_data_percentage = context["valid_data_percentage"]

        # These fields will be initialized on initialize method
        self._file_splits = []
        self._current_batch_index = None
        self._batch_size = None

    def get_total_batch(self):
        return len(self._file_splits)

    def next_batch(self):
        """Return batch if have more splits, else return None"""
        if self._current_batch_index is None:
            raise ValueError("FileInputFormat not initialize yet!")
        elif self._current_batch_index >= len(self._file_splits):
            return None

        train_data = []
        train_target = []
        valid_data = []
        valid_target = []
        training_data_sample_weight = []
        valid_data_sample_weight = []

        line_count = 0
        train_pos_cnt = 0
        train_neg_cnt = 0
        valid_pos_cnt = 0
        valid_neg_cnt = 0
        for line in self.get_split(*self._file_splits[self._current_batch_index]):
            if line_count % 10000 == 0:
                self.tprint("Total loading lines: " + str(line_count))
            columns = line.split(self._delimiter)
            if self._feature_column_nums is None:
                self._feature_column_nums = range(0, len(columns))
                self._feature_column_nums.remove(self._target_index)

            if random.random() >= self._valid_data_percentage:
                # Append training data
                train_target.append([float(columns[self._target_index])])
                if columns[self._target_index] == "1":
                    train_pos_cnt += 1
                else :
                    train_neg_cnt += 1
                single_train_data = []
                for feature_column_num in self._feature_column_nums:
                    single_train_data.append(float(columns[feature_column_num].strip('\n')))
                train_data.append(single_train_data)

                if 0 <= self._sample_weight_column_num < len(columns):
                    weight = float(columns[self._sample_weight_column_num].strip('\n'))
                    if weight < 0.0:
                        self.tprint("Warning: weight is below 0. example:" + line)
                        weight = 1.0
                    training_data_sample_weight.append([weight])
                else:
                    training_data_sample_weight.append([1.0])
            else:
                # Append validation data
                valid_target.append([float(columns[self._target_index])])
                if columns[self._target_index] == "1":
                    valid_pos_cnt += 1
                else:
                    valid_neg_cnt += 1
                single_valid_data = []
                for feature_column_num in self._feature_column_nums:
                    single_valid_data.append(float(columns[feature_column_num].strip('\n')))
                valid_data.append(single_valid_data)

                if 0 <= self._sample_weight_column_num < len(columns):
                    weight = float(columns[self._sample_weight_column_num].strip('\n'))
                    if weight < 0.0:
                        self.tprint("Warning: weight is below 0. example:" + line)
                        weight = 1.0
                    valid_data_sample_weight.append([weight])
                else:
                    valid_data_sample_weight.append([1.0])

        self.tprint("Total data count: " + str(line_count) + ".")
        self.tprint("Train pos count: " + str(train_pos_cnt) + ", neg count: " + str(train_neg_cnt) + ".")
        self.tprint("Valid pos count: " + str(valid_pos_cnt) + ", neg count: " + str(valid_neg_cnt) + ".")
        self._current_batch_index += 1
        return train_data, train_target, valid_data, valid_target, training_data_sample_weight, valid_data_sample_weight

    def initialize(self):
        all_files = gfile.ListDirectory(self._root_folder)
        norm_files = filter(lambda x: not x.startswith(".") and not x.startswith("_"), all_files)
        self.tprint(norm_files)
        self.tprint("Total input file count is " + str(len(norm_files)) + ".")
        sys.stdout.flush()

        file_count = 0
        line_count = 0
        for normal_file in norm_files:
            self.tprint("Now loading " + normal_file + " Progress: " + str(file_count) + "/" + str(len(norm_files)) +
                        ".")
            file_line_cnt, file_splits = self.__split_file(os.path.join(self._root_folder, normal_file))
            sys.stdout.flush()
            file_count += 1
            line_count += file_line_cnt
            self._file_splits.extend(file_splits)
        self.tprint("Total data files: " + str(file_count) + ".")
        self.tprint("Total data count: " + str(line_count) + ".")
        sys.stdout.flush()

        # Set batch size value according to total data count and batch number
        self._batch_size = int(line_count / self._batch_num)
        self._current_batch_index = 0
        if len(norm_files) > 0:
            first_line = self.get_first_line(norm_files)
            columns = first_line.split(self._delimiter)
            if self._feature_column_nums is None:
                self._feature_column_nums = range(0, len(columns))
                self._feature_column_nums.remove(self._target_index)
        self._context["feature_count"] = len(self._feature_column_nums)

    def __split_file(self, file_path):
        splits = []
        total_count = self.get_file_length(file_path)
        batch_number = int(math.ceil(total_count * 1.0 / self._batch_size))
        for i in range(batch_number):
            start = i * self._batch_size
            stop = total_count if (i + 1) * self._batch_size > total_count else (i + 1) * self._batch_size
            splits.append((file_path, start, stop))
        return total_count, splits

