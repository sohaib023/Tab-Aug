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

from augmentor import Augmentor
from pytruth.Document import Document

def apply_ocr(path, image):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    else:
        w, h = image.size
        r = 2500 / w
        image = image.resize((2500, int(r * h)))

        print("OCR start")
        ocr = pytesseract.image_to_data(image,
                                        output_type=pytesseract.Output.DICT,
                                        config="--oem 1")
        print("OCR end")
                    
        bboxes = []
        for i in range(len(ocr['conf'])):
            if ocr['level'][i] > 4 and ocr['text'][i].strip()!="":
                bboxes.append([
                    len(ocr['text'][i]),
                    ocr['text'][i],
                    int(ocr['left'][i] / r),
                    int(ocr['top'][i] / r),
                    int(ocr['left'][i] / r) + int(ocr['width'][i] / r),
                    int(ocr['top'][i] / r) + int(ocr['height'][i] / r)
                    ])
    
        bboxes = sorted(bboxes, key=lambda box: (box[4] - box[2]) * (box[5] - box[3]), reverse=True)
        threshold = np.average([(box[4] - box[2]) * (box[5] - box[3]) for box in bboxes[len(bboxes) // 20: -len(bboxes) // 4]])
        bboxes = [box for box in bboxes if (box[4] - box[2]) * (box[5] - box[3]) < threshold * 30]

        with open(path, "wb") as f:
            pickle.dump(bboxes, f)

        return bboxes

def augment_blocks(table, is_row):
    size = len(table.t.gtCells if is_row else table.t.gtCells[0])

    i = random.choice(range(1, size))
    j = i

    j = random.choice(range(1, size + 1))

    table.replicate_row(i, j)

def remove_blocks(table, is_row):
    return
    blocks = table.rows if is_row else table.columns

    if len(blocks) < 4:
        return False

    i = random.choice(range(1, len(blocks) - 1))
    table.remove(i, is_row=is_row)

def augment_table(augmentor, org_shape, do_col_augmentation=True):
    crop_shape = augmentor.image.shape

    # augmentor.visualize("input")

    column_rules = [
        (8, (0, 1), (0, 2)),
        (5, (0, 2), (0, 2)),
        (3, (1, 2), (0, 1)),
    ] 

    row_rules = [
        (14, (0, 2), (1, 4)),
        (9, (1, 4), (1, 3)),
        (5, (1, 3), (1, 2)),
        (3, (0, 2), (0, 1)),
        (2, (0, 1), (0, 0)),
    ]
    # ^ (min_size, (replicate_range), (remove_range))

    edited = False


    for col, replicate_range, remove_range in column_rules:
        if len(augmentor.t.gtCells[0]) >= col:

            num_add = random.randint(*replicate_range)
            counter = 0
            while counter < num_add:
                size = len(augmentor.t.gtCells[0])
                i = random.choice(range(1, size))
                j = random.choice(range(1, size + 1))
                edited |= augmentor.replicate_column(i, j)

                counter += max(1, len(augmentor.t.gtCells[0]) - size)
                
            num_remove = random.randint(*remove_range)
            counter = 0
            while counter < num_remove:
                size = len(augmentor.t.gtCells[0])
                i = random.choice(range(1, size))
                edited |= augmentor.remove_column(i)

                counter += max(1, size - len(augmentor.t.gtCells[0]))
            break

    for row, replicate_range, remove_range in row_rules:
        if len(augmentor.t.gtCells) >= row:

            num_add = random.randint(*replicate_range)
            counter = 0
            while counter < num_add:
                size = len(augmentor.t.gtCells)
                i = random.choice(range(1, size))
                j = random.choice(range(1, size + 1))
                edited |= augmentor.replicate_row(i, j)

                counter += max(1, len(augmentor.t.gtCells) - size)

            num_remove = random.randint(*remove_range)
            counter = 0
            while counter < num_remove:
                size = len(augmentor.t.gtCells)
                i = random.choice(range(1, size))
                edited |= augmentor.remove_row(i)

                counter += max(1, size - len(augmentor.t.gtCells))
            break

    new_shape = augmentor.image.shape
    # augmentor.visualize("output")
    if new_shape[0] > org_shape[0] or new_shape[1] > org_shape[1]:
        print("Generated image is too large. Discarding sample.")
        return False

    return True

def ensure_exists(filename, log_data):
    if not os.path.exists(filename):
        print("\nWARNING: '", filename, "' not found. Skipping Sample.\n")
        log_data.append("\nWARNING: '" + filename + "' not found. Skipping Sample.\n")
        return False
    else:
        return True


def process_files(image_dir, xml_dir, ocr_dir, out_dir, num_samples, log_file):
    files = list(map(lambda name: os.path.basename(name).rsplit('.', 1)[0], glob.glob(os.path.join(xml_dir,'*.xml'))))

    files.sort()
    log_data = []

    generated = 0
    print(files)
    while generated < num_samples:
        for idx, file in enumerate(files[:10]):
            try:
                if generated >= num_samples:
                    break
                print("[", generated, "/", num_samples, "]", file)

                image_file = os.path.join(image_dir, file + '.png')
                xml_file = os.path.join(xml_dir, file + '.xml')
                ocr_file = os.path.join(ocr_dir, file + '.pkl')

                if not(ensure_exists(image_file, log_data) and ensure_exists(xml_file, log_data)):
                    continue

                img = cv2.imread(image_file)

                ocr = apply_ocr(ocr_file, Image.fromarray(img))
                
                ocr_mask = np.zeros_like(img)
                for word in ocr:
                    txt = word[1].translate(str.maketrans('', '', string.punctuation))
                    if len(txt.strip()) > 0:
                        cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)

                if os.path.exists(image_file) and os.path.exists(xml_file) and os.path.exists(ocr_file):
                    doc = Document(xml_file)
                    for i, table in enumerate(doc.tables):
                        if generated >= num_samples:
                            break

                        augmentor = Augmentor(table, img, ocr)

                        if not augment_table(augmentor, img.shape):
                            continue
                            
                        counter = 0
                        table_name = file + '_' + str(i) + '_' + str(counter)
                        while os.path.exists(os.path.join(out_dir, "images", table_name + '.png')):
                            counter += 1
                            table_name = file + '_' + str(i) + '_' + str(counter)
                            
                        doc = Document()
                        doc.tables.append(augmentor.t)
                        doc.input_file = table_name + '.png'
                        doc.write_to(os.path.join(out_dir, "gt", table_name + '.xml'))

                        cv2.imwrite(os.path.join(out_dir, "images", table_name + '.png'), augmentor.image)
                        with open(os.path.join(out_dir, "ocr", table_name + '.pkl'), 'wb') as f:
                            pickle.dump(augmentor.ocr, f)

                        generated += 1
            except Exception as e:
                log_data.append("Exception thrown: " + str(e))
                print("Exception thrown: ", e)

    if log_file is not None:
        print("\nError logs have been written to: ", log_file)
        with open(log_file, "w") as f:
            for item in log_data:
                f.write("\n" + item + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-img", "--image_dir", type=str, required=True,
                        help="Directory for images")  #, default="data/splits/train_images")

    parser.add_argument("-xml", "--xml_dir", type=str, required=True,
                        help="Directory for xmls")  #, default="data/splits/train_xmls")

    parser.add_argument("-ocr", "--ocr_dir", type=str, required=True,
                        help="Directory for ocr files. (If an OCR file is not found, it will be generated and saved in this directory for future use)")  #, default="data/splits/train_ocr")

    parser.add_argument("-n", "--num_samples", type=int, required=True,
                        help="Number of augmented samples to generate") 

    parser.add_argument("-o", "--out_dir", type=str, required=True,
                        help="Output directory for generated data")  #, default="data/augmented/")

    parser.add_argument("-log", "--log_file", type=str, required=False,
                        help="Output file path for error logging.")


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'ocr'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'gt'), exist_ok=True)
    
    process_files(args.image_dir, args.xml_dir, args.ocr_dir, args.out_dir, args.num_samples, args.log_file)