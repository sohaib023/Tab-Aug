"""THIS MODULE PROVIDES ALL THE UTILITY FUNCTIONS TO THE MAIN MODULES"""

import os
import numpy as np
import datetime
import pickle


def directory_maker(directory_list):
    for directory in directory_list:
        if os.path.exists(directory):
            continue
        else:
            os.mkdir(directory)


def log_errors(_list, logging_path):
    file_version = str(datetime.datetime.now()).replace(" ", "_")
    logging_string = "Total Errors: " + str(len(_list))
    logging_file_path = os.path.join(logging_path, file_version + "_error_log.txt")
    for _, _item in enumerate(_list):
        logging_string += "\n" + str(_item)

    with open(logging_file_path, "w") as _file:
        _file.write(logging_string)
    print("Errors Logged Saved at: ", logging_file_path)


def write_distributions(_tuple, metadata_dir, data_name):
    filename = os.path.join(metadata_dir, data_name + "_metadata.pkl")
    with open(filename, "wb") as f:
        pickle.dump(_tuple, f)


def read_distribution(db_file):
    with open(db_file, "rb") as f:
        (
            row_column_db,
            typeMerge_db,
            prob_merge,
            rowTocolumn_merge_db,
            mergingRows_db,
            mergingColumns_db,
            cell_word_db,
            cell_width_db,
            cell_height_db,
            lined_table_prob,
            lined_table_db,
        ) = pickle.load(f)

    return (
        row_column_db,
        typeMerge_db,
        prob_merge,
        rowTocolumn_merge_db,
        mergingRows_db,
        mergingColumns_db,
        cell_word_db,
        cell_width_db,
        cell_height_db,
        lined_table_prob,
        lined_table_db,
    )


def file_writer(data, filename, path):
    filename = os.path.join(path, filename)
    with open(filename, "w") as f:
        f.write(data)
        f.close()