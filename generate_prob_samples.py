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

from augmentation.augmentor import Augmentor, apply_action

class ProbBasedAugmentor:
    def __init__(self, distribution_filepath, nodes_filepath):
        self.col_steps = [0, 4, 6, 9]
        self.row_steps = [0, 4, 6, 9, 13]
        
        print("Reading:  Augmentation Data...")
        self.distribution = self.read_distributions(distribution_filepath)
        self.file_to_nodes, self.file_to_probs = self.read_nodes(nodes_filepath)
        print("Complete: Augmentation Data.")

    def read_distributions(self, distribution_filepath):
        with open(distribution_filepath, "rb") as f:
            distributions = pickle.load(f)

        rc_categorized = np.zeros((len(self.row_steps), len(self.col_steps)))
        rc_distribution = distributions[0]

        for i, (row_start, row_end) in enumerate(zip(self.row_steps[:-1], self.row_steps[1:])):
            for j, (col_start, col_end) in enumerate(zip(self.col_steps[:-1], self.col_steps[1:])):
                rc_categorized[i, j] = rc_distribution[row_start: row_end, col_start:col_end].sum()
            rc_categorized[i, -1] = rc_distribution[row_start: row_end, self.col_steps[-1]:].sum()

        for j, (col_start, col_end) in enumerate(zip(self.col_steps[:-1], self.col_steps[1:])):
            rc_categorized[-1, j] = rc_distribution[self.row_steps[-1]:, col_start:col_end].sum()

        rc_categorized[-1, -1] = rc_distribution[self.row_steps[-1]:, self.col_steps[-1]:].sum()
        rc_categorized += 0.01

        return rc_categorized

    def find_bin(self, seperators, val):
        for i, sep in enumerate(seperators):
            if val < sep:
                return i - 1
        return len(seperators) - 1

    def read_nodes(self, nodes_filepath):
        with open(nodes_filepath, "rb") as f:
            file_to_nodes = pickle.load(f)

        file_to_probs = {}
        for filename in file_to_nodes.keys():
            categorized = [[[] for i in range(len(self.col_steps))] for j in range(len(self.row_steps))]

            gauss = None

            for idx, node in enumerate(file_to_nodes[filename]):
                r_idx = self.find_bin(self.row_steps, node[0]['h'])
                c_idx = self.find_bin(self.col_steps, node[0]['w'])

                categorized[r_idx][c_idx].append(node)

            freqs = np.array(list(map(lambda x: list(map(len, x)), categorized)))

            probs = self.distribution * freqs
            probs = probs / probs.sum()
            file_to_nodes[filename] = categorized
            file_to_probs[filename] = probs
        return file_to_nodes, file_to_probs

    def generate_gauss(self, center, shape=(5, 5)):
        gauss = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                gauss[i, j] = 1 / (2 ** max(0, abs((i - center[0])) + abs((j - center[1])) - 1))
        return gauss

    def apply_augmentation(self, filename, table, img, ocr):
        augmentor = Augmentor(table, img, ocr)

        r_idx = self.find_bin(self.row_steps, len(table.gtCells))
        c_idx = self.find_bin(self.col_steps, len(table.gtCells[0]))
        gauss = self.generate_gauss((r_idx, c_idx), shape=(len(self.row_steps), len(self.col_steps)))
        
        nodes = self.file_to_nodes[filename]
        probs = self.file_to_probs[filename] * gauss
        probs = probs / probs.sum()

        # Select which size to choose from.
        i = np.random.choice(np.arange(probs.size), p=probs.ravel())
        indices = np.unravel_index(i, probs.shape)

        chosen_node = random.choice(nodes[indices[0]][indices[1]])

        for action in chosen_node[1]:
            return_val = apply_action(augmentor, action)
            assert return_val

        table, img, ocr = augmentor.t, augmentor.image, augmentor.ocr
        assert len(table.gtCells)==chosen_node[0]['h']
        assert len(table.gtCells[0])==chosen_node[0]['w']

        return table, img, ocr