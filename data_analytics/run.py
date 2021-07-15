"""RUNS THE MAIN PIPELINE FOR GENERATING SYNTHETIC LATEX TABLES"""

import os
import glob
import pickle
import datetime
import argparse
import utility as ut
from tqdm import tqdm
import generate_analytics as analytics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-ocr", "--ocr_dir", type=str, required=True, help="path to ocr directory")

    parser.add_argument("-xml", "--xml_dir", type=str, required=True, help="path to xml directory")

    parser.add_argument(
        "-img", "--img_dir", type=str, required=True, help="path to image directory, images should be in PNG"
    )

    parser.add_argument(
        "-m",
        "--metadata_dir",
        type=str,
        required=False,
        help="path to folder where the metadata should be stored",
    )

    parser.add_argument(
        "-l", "--log_dir", type=str, required=False, help="path to folder where the log data would be stored"
    )

    args = parser.parse_args()
    xml_dir = args.xml_dir
    ocr_dir = args.ocr_dir
    img_dir = args.img_dir
    metadata_dir = args.metadata_dir

    if args.log_dir:
        log_dir = args.log_dir
    else:
        log_dir = os.path.join(os.getcwd(), "error_logs")

    directory_list = []
    directory_list.append(metadata_dir)
    directory_list.append(log_dir)
    ut.directory_maker(directory_list)

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
        lined_table_db,
    ) = analytics.get_probability_distribution(xml_dir, ocr_dir, img_dir, log_dir, metadata_dir)
    print(lined_table_db)