# Dataset Augmentation
Use ***augment_data.py*** script for augmenting a dataset. You have to provide folders containing images and their corresponding XML and OCR files. If OCR files are not available you may provide an empty directory and the program will generate the missing  OCR files and save them there. ***log_file*** flag can be used for logging out warnings and errors to a text file. If a file name is not provided the logging will be skipped.

***Note: that the script will not over-write generated files in any case. Thus you can call the augmentation script multiple times into the same output directory without a concern for them being over-written.***

### Usage
```
usage: augment_data.py [-h] -img IMAGE_DIR -xml XML_DIR -ocr OCR_DIR -n
                       NUM_SAMPLES -o OUT_DIR

optional arguments:
  -h, --help            show this help message and exit
  -img IMAGE_DIR, --image_dir IMAGE_DIR
                        Directory for images
  -xml XML_DIR, --xml_dir XML_DIR
                        Directory for xmls
  -ocr OCR_DIR, --ocr_dir OCR_DIR
                        Directory for ocr files. (If an OCR file is not found,
                        it will be generated and saved in this directory for
                        future use)
  -n NUM_SAMPLES, --num_samples NUM_SAMPLES
                        Number of augmented samples to generate
  -o OUT_DIR, --out_dir OUT_DIR
                        Output directory for generated data
  -log LOG_FILE, --log_file LOG_FILE
                        Output file path for error logging.
```

Command: `python augment_data.py -img data/images/ -xml data/xmls/ -ocr data/ocr/ -n 100 -o augmented_data/ -log error_logs.txt`

# TFRecords Generation
Use ***generate_tf_records.py*** script for converting a dataset consisting of images, xmls and ocr data into tfrecord files. 
(If provided ocr directory does not contain required file, it will be generated at run-time and saved in the given directory to save computation in future).

### Usage
```
usage: generate_tf_records.py [-h] -img IMAGE_DIR -ocr OCR_DIR -xml XML_DIR -o
                              OUT_DIR [--filesize FILESIZE] [--visualize]

optional arguments:
  -h, --help            show this help message and exit
  -img IMAGE_DIR, --image_dir IMAGE_DIR
                        Folder containing input images.
  -ocr OCR_DIR, --ocr_dir OCR_DIR
                        Folder containing OCR of input images.
  -xml XML_DIR, --xml_dir XML_DIR
                        Folder containing ground truth xmls.
  -o OUT_DIR, --out_dir OUT_DIR
                        Output path for the generated TFRecords
  --filesize FILESIZE   Number of tables to be stored inside a single TFRecord file.
  --visualize           Visualize generated/parsed data and adjacency matrices
```

Command: `python generate_tf_records.py -img data/images/ -xml data/xmls/ -ocr data/ocr/ -o tfrecords_out/`
