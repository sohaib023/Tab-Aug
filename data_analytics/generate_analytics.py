"""CORE ANALYTICS MODULE: Generates the analytical meta-data from user database"""
import os
import cv2
import glob
import pickle
import numpy as np
import utility as ut
from tqdm import tqdm
from xml.etree import ElementTree

max_rows = 40
max_cols = 17


def merge_check(cell_data):
    """
    ARGUMENTS:

    cell_data: this contains the cell information from a specific table.

    RETURNS:

    merge_metadata: this contains the merging information about different cells.
    """
    merge_metadata = []
    merged_columns = []
    merged_rows = []
    for cell in cell_data:
        if cell.attrib["startCol"] != cell.attrib["endCol"]:
            temp = []
            temp.append(cell.attrib["startCol"])
            temp.append(cell.attrib["endCol"])
            temp.append(cell.attrib["startRow"])
            merged_columns.append(temp)
        if cell.attrib["startRow"] != cell.attrib["endRow"]:
            temp = []
            temp.append(cell.attrib["startRow"])
            temp.append(cell.attrib["endRow"])
            temp.append(cell.attrib["startCol"])
            merged_rows.append(temp)
    merge_metadata.append(merged_rows)
    merge_metadata.append(merged_columns)
    return merge_metadata


def fetch_table_ocr(ocr_data, x0, x1, y0, y1):
    """
    #TODO
    ARGUMENTS:

    ocr_data: This contains the ocr data of the entire file.

    x0: is the x coordinate of the top left of the table.

    x1: is the y coordinate of the top left of the table.

    y0: is the x coordinate of the bottom right of the table.

    y1: is the y coordinate of the bottom right of the table.

    RETURNS:

    table_ocr_data: This return contains the ocr data that is inside
    the tables bounding box.
    """
    table_ocr_data = []
    for data_item in ocr_data:
        if (
            (int(data_item[2]) >= int(x0))
            and (int(data_item[4]) <= int(x1))
            and (int(data_item[3]) >= int(y0))
            and (int(data_item[5]) <= int(y1))
        ):
            table_ocr_data.append(data_item)
    return table_ocr_data


def cell_distributed_ocr(raw_ocr_data, cell_data):
    """
    ARGUMENTS:

    raw_ocr_data:this is the table distributed ocr data that.

    cell_data: this contains the cell information of a table.

    RETURNS:

    final_ocr_data: this returns a list of ocr data relavent according to each cell
    """
    final_ocr_data = []
    for cell in cell_data:
        temp = []
        temp.append(cell)
        for data_item in raw_ocr_data:
            if (
                (int(data_item[2]) >= int(cell.attrib["x0"]))
                and (int(data_item[4]) <= int(cell.attrib["x1"]))
                and (int(data_item[3]) >= int(cell.attrib["y0"]))
                and (int(data_item[5]) <= int(cell.attrib["y1"]))
            ):
                temp.append(data_item)
        final_ocr_data.append(temp)
    return final_ocr_data


def localize_data(xml_file, ocr_file, img_file):
    """
    ARGUMENTS:

    xml_file: a path to an xml file inside a folder.

    ocr_file: a path to an ocr file inside a folder.

    img_file: a path to an image file inside a folder.

    RETURNS:

    analytics_data: a data item containing all the relavent analytics for
    the given xml and ocr file.
    """
    analytics_data = []
    ocr_data = []
    with open(ocr_file, "rb") as f:
        ocr_data = pickle.load(f)
    if os.path.exists(xml_file) and os.path.exists(ocr_file) and os.path.exists(img_file):
        tree = ElementTree.parse(xml_file)
        root = tree.getroot()
        for obj in root.findall(".//Table"):
            data_list = []
            if len(obj.findall("Row")) + 1 >= max_rows:
                continue
            if len(obj.findall("Column")) + 1 >= max_cols:
                continue
            data_list.append(len(obj.findall("Row")) + 1)
            data_list.append(len(obj.findall("Column")) + 1)
            cell_data = obj.findall("Cell")
            data_list.append(len(cell_data))
            merged_data = merge_check(cell_data)
            data_list.append(merged_data)
            data_list.append(obj.attrib["orientation"])
            raw_ocr_data = fetch_table_ocr(
                ocr_data,
                obj.attrib["x0"],
                obj.attrib["x1"],
                obj.attrib["y0"],
                obj.attrib["y1"],
            )
            final_ocr_data = cell_distributed_ocr(raw_ocr_data, cell_data)
            data_list.append(final_ocr_data)
            data_list.append(xml_file)

            rows = [int(obj.attrib["y0"])]
            for row in obj.findall("Row"):
                rows.append(int(row.attrib["y0"]))
            rows.append(int(obj.attrib["y0"]))

            cols = [int(obj.attrib["x0"])]
            for col in obj.findall("Column"):
                cols.append(int(col.attrib["x0"]))
            cols.append(int(obj.attrib["x0"]))

            data_list.append(rows)
            data_list.append(cols)
            data_list.append(img_file)
            analytics_data.append(data_list)
        return analytics_data


def get_lined_table_distribution(analytics_data, x=17, y=17):
    """
    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data.

    RETURNS:

    lined_table_distribution: a 1D matrix containing line distribution in data.

    index 0: contains frequency of fully lined tables

    index 1: contains frequency of partially row lined tables

    index 2: contains frequency of partially column lined tables

    index 3: contains frequency of partially column lined tables

    """

    lined_table_db = np.zeros((4))
    for data_item in tqdm(analytics_data):
        img_path = data_item[9]
        _img = cv2.imread(img_path)

        tmp = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        _img = cv2.adaptiveThreshold(tmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        kernel = np.ones((1, 5), np.uint8)
        horizontal_lines = np.copy(_img)
        horizontal_lines = cv2.dilate(horizontal_lines, kernel, iterations=x)
        horizontal_lines = cv2.erode(horizontal_lines, kernel, iterations=x)
        kernel = np.ones((5, 1), np.uint8)
        vertical_lines = np.copy(_img)
        vertical_lines = cv2.dilate(vertical_lines, kernel, iterations=y)
        vertical_lines = cv2.erode(vertical_lines, kernel, iterations=y)
        ver_pos = np.where(vertical_lines == 0)
        hor_pos = np.where(horizontal_lines == 0)
        _w, _h = _img.shape
        vert_img = np.zeros((_w, _h), dtype=np.uint8)
        hor_img = np.zeros((_w, _h), dtype=np.uint8)
        vert_img[ver_pos] = 255
        hor_img[hor_pos] = 255
        vert_lab, _ = cv2.connectedComponents(vert_img)
        hor_lab, _ = cv2.connectedComponents(hor_img)
        vert_lab = vert_lab - 1
        hor_lab = hor_lab - 1

        if vert_lab > 2 and hor_lab > 2:
            lined_table_db[0] += 1
        elif vert_lab <= 2 and hor_lab > 2:
            lined_table_db[1] += 1
        elif vert_lab > 2 and hor_lab <= 2:
            lined_table_db[2] += 1
        elif vert_lab < 2 and hor_lab < 2:
            lined_table_db[3] += 1
    return lined_table_db


def get_row_column_distribution(analytics_data):
    """
    Gets us the row_column Distribution of tables occurring throughout the dataset
    this gives us an idea on what type of tables are the most commonly occuring ones

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data.

    RETURNS:

    row_column_db: a 2D matrix containing the distribution
    """
    total_tables = len(analytics_data)
    row_column_db = np.zeros((50, 30))
    for data_item in analytics_data:
        row_column_db[int(data_item[0]), int(data_item[1])] += 1
    row_column_db = np.true_divide(row_column_db, total_tables)
    return row_column_db


def get_RowToColumn_merge_db(analytics_data):
    """
    This method gives us the distribution between a row and column merging in
    all types of tables

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURNS:

    a 3D numpy matrix consisting of the above mentioned distribution
    """
    RtC_merge_db = np.zeros((50, 30, 2))
    for data_item in analytics_data:
        if (int(len(data_item[3][0])) != 0) or (int(len(data_item[3][1])) != 0):
            RtC_merge_db[int(data_item[0]), int(data_item[1]), int(0)] += len(data_item[3][0])
            RtC_merge_db[int(data_item[0]), int(data_item[1]), int(1)] += len(data_item[3][1])
    RtC_merge_db[:, :, 0], RtC_merge_db[:, :, 1] = (
        (RtC_merge_db[:, :, 0]) / (RtC_merge_db[:, :, 0] + RtC_merge_db[:, :, 1]),
        (RtC_merge_db[:, :, 1]) / (RtC_merge_db[:, :, 0] + RtC_merge_db[:, :, 1]),
    )
    RtC_merge_db = np.nan_to_num(RtC_merge_db)
    return RtC_merge_db


def get_cell_width_db(analytics_data):
    """
    This method gives us the min, average and max of the width of cells distributed
    throughout the data distinguishable by row and columns in the table.

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURN:

    cell_width_db: a 5D numpy array having distributions of widths over cells in
    different tables. it keeps track of min, avg, max, count in particular table
    types.
    """

    cell_width_db = np.zeros((max_rows, max_cols, max_rows, max_cols, 3), dtype="float")
    for data_item in analytics_data:
        for cell_list in data_item[5]:
            cell_obj = cell_list[0]
            _row = int(data_item[0])
            _col = int(data_item[1])
            _curr_r = int(cell_obj.attrib["startRow"])
            _curr_c = int(cell_obj.attrib["startCol"])
            _width = data_item[8][_curr_c + 1] - data_item[8][_curr_c]

            # changing average
            cell_width_db[_row, _col, _curr_r, _curr_c, 0] += _width
            cell_width_db[_row, _col, _curr_r, _curr_c, 2] += 1
    cell_width_db[:, :, :, :, 0] /= cell_width_db[:, :, :, :, 2]
    for data_item in analytics_data:
        for cell_list in data_item[5]:
            cell_obj = cell_list[0]
            _row = int(data_item[0])
            _col = int(data_item[1])
            _curr_r = int(cell_obj.attrib["startRow"])
            _curr_c = int(cell_obj.attrib["startCol"])
            _width = data_item[8][_curr_c + 1] - data_item[8][_curr_c]

            # changing average
            mean = cell_width_db[_row, _col, _curr_r, _curr_c, 0]
            cell_width_db[_row, _col, _curr_r, _curr_c, 1] += (_width - mean) ** 2
    cell_width_db[:, :, :, :, 1] /= cell_width_db[:, :, :, :, 2]
    cell_width_db[:, :, :, :, 1] = np.sqrt(cell_width_db[:, :, :, :, 1])

    return cell_width_db


def get_cell_height_db(analytics_data):
    """
    This method gives us the min, average and max of the height of cells distributed
    throughout the data distinguishable by row and columns in the table.

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURN:

    cell_height_db: a 5D numpy array having distributions of heights over cells in
    different tables.
    """
    cell_height_db = np.zeros((max_rows, max_cols, max_rows, max_cols, 4), dtype="float")
    for data_item in analytics_data:
        for cell_list in data_item[5]:
            cell_obj = cell_list[0]
            _row = int(data_item[0])
            _col = int(data_item[1])
            _curr_r = int(cell_obj.attrib["startRow"])
            _curr_c = int(cell_obj.attrib["startCol"])
            _height = float(cell_obj.attrib["y1"]) - float(cell_obj.attrib["y0"])

            # checking minimum
            _current_min = cell_height_db[_row, _col, _curr_r, _curr_c, 0]
            if _height < _current_min:
                cell_height_db[_row, _col, _curr_r, _curr_c, 0] = _height

            # checking maximum
            _current_max = cell_height_db[_row, _col, _curr_r, _curr_c, 2]
            if _height > _current_max:
                cell_height_db[_row, _col, _curr_r, _curr_c, 2] = _height

            # changing average
            _old_average = cell_height_db[_row, _col, _curr_r, _curr_c, 1]
            n1 = cell_height_db[_row, _col, _curr_r, _curr_c, 3]
            cell_height_db[_row, _col, _curr_r, _curr_c, 3] += 1
            n2 = cell_height_db[_row, _col, _curr_r, _curr_c, 3]
            _new_average = ((_old_average * n1) + _height) / n2
            cell_height_db[_row, _col, _curr_r, _curr_c, 1] = float(_new_average)
    return cell_height_db


def get_type_merge_db(analytics_data):
    """
    This method gives us the merge distribution by type, meaning the ditribution
    is based on what probability a certain degree table contains a merging

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent dat

    RETURNS:

    row_column_merge_db: a 2D matrix containing the distribution
    """
    typeMerge_db = np.zeros((max_rows, max_cols))
    total_merging_tables = 0
    for data_item in analytics_data:
        if int(len(data_item[3][0])) != 0 or int(len(data_item[3][1])) != 0:
            total_merging_tables += 1
            typeMerge_db[int(data_item[0]), int(data_item[1])] += 1
    if total_merging_tables != 0:
        typeMerge_db = np.true_divide(typeMerge_db, total_merging_tables)
    return typeMerge_db


def get_probability_of_merging(analytics_data):
    """
    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURNS:

    a simple probability of a merge happening in our given data set
    """
    total_tables = len(analytics_data)
    total_merging_tables = 0
    for data_item in analytics_data:
        if int(len(data_item[3][0])) != 0 or int(len(data_item[3][1])) != 0:
            total_merging_tables += 1
    return total_merging_tables / total_tables


def get_mergingRows_db(analytics_data):
    """
    This method gives us a distribution of merging rows in a 5D matrix

    1. index identifies the rows of the table type

    2. index identifies the columns of the table type

    3. index identifies the merging start row

    4. index identifies the merging end row

    5. index identifies the column they are merging on

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURNS:

    row_merge_db: a 5D Matrix containing all the row merging data
    """
    row_merge_db = np.zeros((max_rows, max_cols, max_rows, max_rows, max_cols))
    for data_item in analytics_data:
        for merge_item in data_item[3][0]:
            row_merge_db[
                int(data_item[0]),
                int(data_item[1]),
                int(merge_item[0]),
                int(merge_item[1]),
                int(merge_item[2]),
            ] += 1
    return row_merge_db


def get_mergingColumns_db(analytics_data):
    """
    This method gives us a distribution of merging columns in a 3d matrix

    1. index identifies the rows of the table type

    2. index identifies the columns of the table type

    3. index identifies the merging start column

    4. index identifies the merging end column

    5. index identifies the row they are merging on

    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURNS:

    col_merge_db: a 5D Matrix containing all the column merging data

    """
    col_merge_db = np.zeros((max_rows, max_cols, max_cols, max_cols, max_rows))
    for data_item in analytics_data:
        for merge_item in data_item[3][1]:
            col_merge_db[
                int(data_item[0]),
                int(data_item[1]),
                int(merge_item[0]),
                int(merge_item[1]),
                int(merge_item[2]),
            ] += 1
    return col_merge_db


def get_cell_to_word_distribution(analytics_data):
    """
    ARGUMENTS:

    analytics_data: a list of iterable tables with relavent data

    RETURNS:

    cell_word_db: A 5D matrix containing probability of words per cell differentiated
        by types of tables that occur throughout the dataset
    """
    cell_word_db = np.zeros((max_rows, max_cols, max_rows, max_cols, 70), dtype="float")
    for data_item in analytics_data:
        for cell_item in data_item[5]:
            num_words = int(len(cell_item) - 1)
            word_bin = num_words // 3
            cell_word_db[
                int(data_item[0]),
                int(data_item[1]),
                int(cell_item[0].attrib["startRow"]),
                int(cell_item[0].attrib["startCol"]),
                min(word_bin, 69),
            ] += 1
    return cell_word_db


def read_probability_distribution(distribution_file_path):
    """
    ARGUMENTS:

    distribution_file_path: path to the pickle file containing the distribution.

    RETURNS:

    db_tuple: distribution tuple containing the distribution data

    """
    db_tuple = ut.read_distribution(distribution_file_path)
    return db_tuple


def get_probability_distribution(
    xml_dir,
    ocr_dir,
    img_dir,
    log_dir=os.path.join(os.getcwd(), "error_logs"),
    metadata_dir=os.path.join(os.getcwd(), "metadata"),
    dataset_name="default-icdar",
):
    """
    This method gets us all the distribution matrices of the data

    ARGUMENTS:
        data_path_list: a list of iterable paths where the dataset is stored.

    RETURNS:
        tuple of all the numpy matrices containing all the distributions
    """
    # xml is used as the final file list
    file_list = glob.glob(os.path.join(xml_dir, "*.xml"))

    files_notfound = []
    analytics_data = []
    for _file in tqdm(file_list):
        ocr_file_name = _file.split("/")[-1][:-3] + "pkl"
        img_file_name = _file.split("/")[-1][:-3] + "png"
        ocr_file_path = os.path.join(ocr_dir, ocr_file_name)
        img_file_path = os.path.join(img_dir, img_file_name)
        if os.path.exists(ocr_file_path) and os.path.exists(img_file_path):
            analytics_data += localize_data(_file, ocr_file_path, img_file_path)
        else:
            if not os.path.exists(ocr_file_path) and not os.path.exists(img_file_path):
                files_notfound.append([ocr_file_path, img_file_path])
            elif not os.path.exists(ocr_file_path):
                files_notfound.append([ocr_file_path])
            else:
                files_notfound.append([img_file_path])
    ut.log_errors(files_notfound, log_dir)

    row_column_db = get_row_column_distribution(analytics_data)
    typeMerge_db = get_type_merge_db(analytics_data)
    prob_merge = get_probability_of_merging(analytics_data)
    rowTocolumn_merge_db = get_RowToColumn_merge_db(analytics_data)
    mergingRows_db = get_mergingRows_db(analytics_data)
    mergingColumns_db = get_mergingColumns_db(analytics_data)
    cell_word_db = get_cell_to_word_distribution(analytics_data)
    cell_width_db = get_cell_width_db(analytics_data)
    cell_height_db = get_cell_height_db(analytics_data)
    lined_table_db = get_lined_table_distribution(analytics_data)

    distribution_tuple = (
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
    )

    ut.write_distributions(distribution_tuple, metadata_dir, dataset_name)

    return distribution_tuple
