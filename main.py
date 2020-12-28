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

from augmentor import augment_table
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

def ensure_exists(filename, log_data):
    if not os.path.exists(filename):
        print("\nWARNING: '", filename, "' not found. Skipping Sample.\n")
        log_data.append("WARNING: '" + filename + "' not found. Skipping Sample.")
        return False
    else:
        return True

def process_files(image_dir, xml_dir, ocr_dir, out_dir, num_samples, log_file, visualize):
    files = list(map(lambda name: os.path.basename(name).rsplit('.', 1)[0], glob.glob(os.path.join(xml_dir,'*.xml'))))

    files.sort()
    log_data = []

    generated = 0
    while generated < num_samples:
        for idx, file in enumerate(files):
            try:
                if generated >= num_samples:
                    break
                print("[", generated, "/", num_samples, "]", file)

                image_file = os.path.join(image_dir, file + '.png')
                xml_file = os.path.join(xml_dir, file + '.xml')
                ocr_file = os.path.join(ocr_dir, file + '.pkl')

                if not(ensure_exists(image_file, log_data) and ensure_exists(xml_file, log_data)):
                    continue

                doc_img = cv2.imread(image_file)

                doc_ocr = apply_ocr(ocr_file, Image.fromarray(doc_img))
                
                # ocr_mask = np.zeros_like(doc_img)
                # for word in doc_ocr:
                #     txt = word[1].translate(str.maketrans('', '', string.punctuation))
                #     if len(txt.strip()) > 0:
                #         cv2.rectangle(ocr_mask, (word[2], word[3]), (word[4], word[5]), 255, -1)

                if os.path.exists(image_file) and os.path.exists(xml_file) and os.path.exists(ocr_file):
                    doc = Document(xml_file)
                    for i, table in enumerate(doc.tables):
                        if generated >= num_samples:
                            break

                        table, img, ocr = augment_table(table, doc_img, doc_ocr)
                            
                        counter = 0
                        table_name = file + '_' + str(i) + '_' + str(counter)
                        while os.path.exists(os.path.join(out_dir, "images", table_name + '.png')):
                            counter += 1
                            table_name = file + '_' + str(i) + '_' + str(counter)
                            
                        doc = Document()
                        doc.tables.append(table)
                        doc.input_file = table_name + '.png'
                        doc.write_to(os.path.join(out_dir, "gt", table_name + '.xml'))

                        cv2.imwrite(os.path.join(out_dir, "images", table_name + '.png'), img)
                        with open(os.path.join(out_dir, "ocr", table_name + '.pkl'), 'wb') as f:
                            pickle.dump(ocr, f)

                        if visualize:
                            table.visualize(img)
                            cv2.imwrite(os.path.join(args.out_dir,'vis', table_name + '.png'), img)
                        generated += 1
            except Exception as e:
                log_data.append("Exception thrown: " + str(e))
                print("Exception thrown: ", e)

    if log_file is not None:
        print("\nError logs have been written to: ", log_file)
        with open(log_file, "w") as f:
            for item in log_data:
                f.write(item + "\n")

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

    parser.add_argument("-vis", "--visualize", action="store_true")


    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'images'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'ocr'), exist_ok=True)
    os.makedirs(os.path.join(args.out_dir,'gt'), exist_ok=True)

    if args.visualize:
        os.makedirs(os.path.join(args.out_dir,'vis'), exist_ok=True)
    
    os.makedirs(args.ocr_dir, exist_ok=True)

    process_files(args.image_dir, args.xml_dir, args.ocr_dir, args.out_dir, args.num_samples, args.log_file, args.visualize)