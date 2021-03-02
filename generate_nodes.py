import os
import glob
import pickle
import shutil
import string
import random
import argparse
import pytesseract

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

from xml.dom import minidom
from xml.etree import ElementTree as ET

# from augmentor import augment_table
from tree_explorer import dfs_wrapper
from truthpy import Document

def are_equal(table1, table2):
    try:
        assert table1 == table2
        assert len(table1.gtRows) == len(table2.gtRows)
        assert len(table1.gtCols) == len(table2.gtCols)

        for row1, row2 in zip(table1.gtCells, table2.gtCells):
            for cell1, cell2 in zip(row1, row2):
                assert cell1 == cell2
                assert cell1.startRow == cell2.startRow
                assert cell1.startCol == cell2.startCol
                assert cell1.endRow == cell2.endRow
                assert cell1.endCol == cell2.endCol
    except Exception as e:
        return False
    # print("Removed")
    return True

def eliminate_matching(all_nodes):
    i = 0
    while i < len(all_nodes):
        j = i + 1
        while j < len(all_nodes):
            if i != j and are_equal(all_nodes[i][0], all_nodes[j][0]):
                all_nodes.remove(all_nodes[j])
                j -= 1
            j += 1
        i += 1

    return all_nodes

def process_files(xml_dir, log_file):
    files = list(map(lambda name: os.path.basename(name).rsplit('.', 1)[0], glob.glob(os.path.join(xml_dir,'*.xml'))))

    files.sort()
    log_data = []

    # while generated < num_samples:
    all_nodes_samples = {}
    for idx, file in enumerate(files):
        # try:
        print("[", idx, "/", len(files), "]", file)

        xml_file = os.path.join(xml_dir, file + '.xml')

        if os.path.exists(xml_file):
            doc = Document(xml_file)
            for i, table in enumerate(doc.tables):

                all_nodes = dfs_wrapper(table)
                print("\t", len(all_nodes))
                all_nodes = [x[1:] for x in all_nodes]

                all_nodes_samples[file] = all_nodes

    with open("all_nodes.pkl", "wb") as f:
    	pickle.dump(all_nodes_samples, f)

    if log_file is not None:
        print("\nError logs have been written to: ", log_file)
        with open(log_file, "w") as f:
            for item in log_data:
                f.write(item + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-xml", "--xml_dir", type=str, required=True,
                        help="Directory for xmls")  #, default="data/splits/train_xmls")

    parser.add_argument("-log", "--log_file", type=str, required=False,
                        help="Output file path for error logging.")

    args = parser.parse_args()
    
    process_files(args.xml_dir, args.log_file)
