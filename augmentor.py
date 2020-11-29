import cv2
import copy
import random
import numpy as np

from xml.dom import minidom
from xml.etree import ElementTree as ET

from pytruth.GTElement import GTElement, Row, Column
from pytruth.Table import Table

def translate_ocr(ocr, translation):
    x_trans, y_trans = translation
    for i in range(len(ocr)):
        ocr[i][2] += x_trans
        ocr[i][3] += y_trans
        ocr[i][4] += x_trans
        ocr[i][5] += y_trans
    return ocr

def get_bounded_ocr(ocr, p1, p2, remove_org=False, threshold=0.5):
    x1, y1 = p1
    x2, y2 = p2
    bounded_ocr = []

    for i in range(len(ocr)):
        _x1 = max(ocr[i][2], x1)    
        _y1 = max(ocr[i][3], y1)
        _x2 = min(ocr[i][4], x2)
        _y2 = min(ocr[i][5], y2)

        if _x1 < _x2 and _y1 < _y2:
            w = ocr[i][4] - ocr[i][2]
            h = ocr[i][5] - ocr[i][3]

            area_intersection = (_x2 - _x1) * (_y2 - _y1)
            area_word = w*h
            if area_intersection / area_word > threshold:
                bounded_ocr.append(ocr[i].copy())

    if remove_org:
        ocr[:] = [word for word in ocr if word not in bounded_ocr]

    return bounded_ocr

class Augmentor:
    def __init__(self, table, doc_image, doc_ocr):
        self.t = table
        
        self.doc_shape = doc_image.shape[:2]
        self.image = doc_image[self.t.y1:self.t.y2, self.t.x1:self.t.x2]

        self.ocr = translate_ocr(get_bounded_ocr(doc_ocr, (self.t.x1, self.t.y1), (self.t.x2, self.t.y2)), (-self.t.x1, -self.t.y1))

        self.t.move(-self.t.x1, -self.t.y1)

    def get_column_range(self, col):
        if col == len(self.t.gtCells[0]):
            return True, col, col, None
        column_bbox = GTElement(self.t.gtCells[0][col].x1, self.t.y1, self.t.gtCells[0][col].x2, self.t.y2)

        filtered = list(filter(lambda s: (column_bbox & s).area() > 1e-2, self.t.gtSpans))

        if len(filtered) == 0:
            return True, col, col, []

        min_x = min(map(lambda s: s.x1, filtered))
        max_x = max(map(lambda s: s.x2, filtered))

        cell1 = self.t.getCellAtPoint((min_x, 1))
        cell2 = self.t.getCellAtPoint((max_x, 1))

        column_bbox = GTElement(cell1.x1, self.t.y1, cell2.x2, self.t.y2)

        filtered_partial = list(filter(lambda s: (column_bbox & s).area() > 0, self.t.gtSpans))
        filtered_full = list(filter(lambda s: (column_bbox & s).area() == s.area(), self.t.gtSpans))

        is_convex = len(filtered_partial) == len(filtered_full)

        return is_convex, cell1.startCol, cell2.endCol, filtered_full

    def get_row_range(self, row):
        if row == len(self.t.gtCells):
            return True, row, row, None
        row_bbox = GTElement(self.t.x1, self.t.gtCells[row][0].y1, self.t.x2, self.t.gtCells[row][0].y2)

        filtered = list(filter(lambda s: (row_bbox & s).area() > 1e-2, self.t.gtSpans))

        if len(filtered) == 0:
            return True, row, row, []

        min_y = min(map(lambda s: s.y1, filtered))
        max_y = max(map(lambda s: s.y2, filtered))

        cell1 = self.t.getCellAtPoint((1, min_y))
        cell2 = self.t.getCellAtPoint((1, max_y))

        row_bbox = GTElement(self.t.x1, cell1.y1, self.t.x2, cell2.y2)

        filtered_partial = list(filter(lambda s: (row_bbox & s).area() > 0, self.t.gtSpans))
        filtered_full = list(filter(lambda s: (row_bbox & s).area() == s.area(), self.t.gtSpans))

        is_convex = len(filtered_partial) == len(filtered_full)
            
        return is_convex, cell1.startRow, cell2.endRow, filtered_full

    def replicate_row(self, idx_copy, idx_paste):
        is_convex, idx_c1, idx_c2, spans = self.get_row_range(idx_copy)
        is_convex2, idx_p1, idx_p2, _ = self.get_row_range(idx_paste)

        if not is_convex or not is_convex2:
            print("Not convex")
            return False

        if abs(idx_p1 - idx_paste) <= abs(idx_p2 - idx_paste):
            idx_paste = idx_p1
        else:
            idx_paste = idx_p2 + 1

        copy_y1 = self.t.gtCells[idx_c1][0].y1
        copy_y2 = self.t.gtCells[idx_c2][0].y2
        paste_y1 = self.t.gtCells[idx_paste - 1][0].y2
        h = copy_y2 - copy_y1

        image_row = self.image[copy_y1:copy_y2, :]

        new_shape = list(self.image.shape)
        new_shape[0] += image_row.shape[0]

        image_new = np.zeros((new_shape), dtype=self.image.dtype)
        image_new[:paste_y1, :]             = self.image[:paste_y1, :]
        image_new[paste_y1:paste_y1 + h, :] = image_row
        image_new[paste_y1 + h:, :]         = self.image[paste_y1:, :]
        self.image = image_new

        ocr_row = get_bounded_ocr(self.ocr, (0, copy_y1), (self.t.x2, copy_y2))
        ocr_row = translate_ocr(ocr_row, (0, paste_y1 - copy_y1))

        self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (0, paste_y1), (self.t.x2, self.t.y2), remove_org=True), (0, h))
        self.ocr += ocr_row

        rows = [Row(self.t.x1, self.t.y1, self.t.x2)] + self.t.gtRows

        rows_add = list(map(lambda x: x.move_im(0, paste_y1 - copy_y1), rows[idx_c1:idx_c2 + 1]))

        rows[idx_paste:] = [row.move_im(0, h) for row in rows[idx_paste:]]
        rows += rows_add
        rows.sort(key=lambda x: x.y1)
        self.t.gtRows = rows[1:]

        spans_add = list(map(lambda x: x.move_im(0, paste_y1 - copy_y1), spans))

        for span in filter(lambda x: x.y1 > paste_y1, self.t.gtSpans):
            span.move(0, h)
        self.t.gtSpans += spans_add
        self.t.y2 += h

        self.t.evaluateCells()

        return True

    def remove_row(self, idx):
        is_convex, idx_1, idx_2, spans = self.get_row_range(idx)

        if not is_convex:
            print("Not convex")
            return False

        y1 = self.t.gtCells[idx_1][0].y1
        y2 = self.t.gtCells[idx_2][0].y2
        h = y2 - y1

        if h >= self.image.shape[0] * 0.6 or len(self.t.gtCells) - (idx_1 - idx_2 + 1) <= 3:
            return False

        new_shape = list(self.image.shape)
        new_shape[0] -= h

        image_new = np.zeros((new_shape), dtype=self.image.dtype)
        image_new[:y1, :]             = self.image[:y1, :]
        image_new[y1:, :]         = self.image[y2:, :]
        self.image = image_new

        ocr_row = get_bounded_ocr(self.ocr, (0, y1), (self.t.x2, y2), remove_org=True)

        self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (0, y1), (self.t.x2, self.t.y2), remove_org=True), (0, y1 - y2))

        rows = [Row(self.t.x1, self.t.y1, self.t.x2)] + self.t.gtRows
        rows = [row for row in rows if row not in rows[idx_1: idx_2 + 1]]

        rows[idx_1:] = [row.move_im(0, -h) for row in rows[idx_1:]]
        rows.sort(key=lambda x: x.y1)
        self.t.gtRows = rows[1:]

        self.t.gtSpans = [span for span in self.t.gtSpans if span not in spans]

        for span in filter(lambda x: x.y1 > y1, self.t.gtSpans):
            span.move(0, -h)
        self.t.y2 -= h

        self.t.evaluateCells()

        return True

    def replicate_column(self, idx_copy, idx_paste):
        is_convex, idx_c1, idx_c2, spans = self.get_column_range(idx_copy)
        is_convex2, idx_p1, idx_p2, _ = self.get_column_range(idx_paste)

        if not is_convex or not is_convex2:
            print("Not convex")
            return False

        if abs(idx_p1 - idx_paste) <= abs(idx_p2 - idx_paste):
            idx_paste = idx_p1
        else:
            idx_paste = idx_p2 + 1

        copy_x1 = self.t.gtCells[0][idx_c1].x1
        copy_x2 = self.t.gtCells[0][idx_c2].x2
        paste_x1 = self.t.gtCells[0][idx_paste - 1].x2
        w = copy_x2 - copy_x1

        if self.image.shape[1] + w > self.doc_shape[1]:
            # print("too big", self.image.shape[1] ,'+', w, self.doc_shape[1])
            return False

        image_col = self.image[:, copy_x1:copy_x2]

        new_shape = list(self.image.shape)
        new_shape[1] += image_col.shape[1]

        image_new = np.zeros((new_shape), dtype=self.image.dtype)
        image_new[:, :paste_x1]             = self.image[:, :paste_x1]
        image_new[:, paste_x1:paste_x1 + w] = image_col
        image_new[:, paste_x1 + w:]         = self.image[:, paste_x1:]
        self.image = image_new

        ocr_col = get_bounded_ocr(self.ocr, (copy_x1, 0), (copy_x2, self.t.y2))
        ocr_col = translate_ocr(ocr_col, (0, paste_x1 - copy_x1))

        self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (paste_x1, 0), (self.t.x2, self.t.y2), remove_org=True), (w, 0))
        self.ocr += ocr_col

        cols = [Column(self.t.x1, self.t.y1, self.t.y2)] + self.t.gtCols

        cols_add = list(map(lambda x: x.move_im(paste_x1 - copy_x1, 0), cols[idx_c1:idx_c2 + 1]))

        cols[idx_paste:] = [col.move_im(w, 0) for col in cols[idx_paste:]]
        cols += cols_add
        cols.sort(key=lambda x: x.x1)
        self.t.gtCols = cols[1:]

        spans_add = list(map(lambda x: x.move_im(paste_x1 - copy_x1, 0), spans))

        for span in filter(lambda x: x.x1 > paste_x1, self.t.gtSpans):
            span.move(w, 0)
        self.t.gtSpans += spans_add
        self.t.x2 += w

        self.t.evaluateCells()
        return True

    def remove_column(self, idx):
        is_convex, idx_1, idx_2, spans = self.get_column_range(idx)

        if not is_convex:
            print("Not convex")
            return False

        x1 = self.t.gtCells[0][idx_1].x1
        x2 = self.t.gtCells[0][idx_2].x2
        w = x2 - x1

        if w >= self.image.shape[1] * 0.7 or len(self.t.gtCells[0]) - (idx_1 - idx_2 + 1) <= 2:
            return False

        new_shape = list(self.image.shape)
        new_shape[1] -= w

        image_new = np.zeros((new_shape), dtype=self.image.dtype)
        image_new[:, :x1] = self.image[:, :x1]
        image_new[:, x1:] = self.image[:, x2:]
        self.image = image_new

        ocr_col = get_bounded_ocr(self.ocr, (x1, 0), (x2, self.t.y2), remove_org=True)

        self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (x1, 0), (self.t.x2, self.t.y2), remove_org=True), (0, x1 - x2))


        cols = [Column(self.t.x1, self.t.y1, self.t.y2)] + self.t.gtCols
        cols = [col for col in cols if col not in cols[idx_1: idx_2 + 1]]

        cols[idx_1:] = [col.move_im(-w, 0) for col in cols[idx_1:]]

        cols.sort(key=lambda x: x.x1)
        self.t.gtCols = cols[1:]

        self.t.gtSpans = [span for span in self.t.gtSpans if span not in spans]

        for span in filter(lambda x: x.x1 > x1, self.t.gtSpans):
            span.move(-w, 0)
        self.t.x2 -= w

        self.t.evaluateCells()

        return True

    def remove(self, to_remove, is_row):
        if is_row:
            minspan = 0
            maxspan = 0
            for i in range(len(self.cells[to_remove])):
                if self.cells[to_remove][i].attrib['rowspan'] != 0:
                    minspan = min(minspan, self.cells[to_remove][i].attrib['rowspan'])
            minspan *= -1
            to_remove -= minspan
            for i in range(len(self.cells[to_remove])):
                if self.cells[to_remove][i].attrib['rowspan'] != 0:
                    maxspan = max(maxspan, self.cells[to_remove][i].attrib['rowspan'])

            y1 = self.rows[to_remove]
            y2 = self.rows[to_remove + maxspan + 1]

            if y2 - y1 >= self.image.shape[0] or len(self.rows) - (maxspan + 1) < 3:
                return False

            image_row = self.image[y1:y2, :]

            image_new = np.zeros((self.image.shape[0] - image_row.shape[0], self.image.shape[1]), dtype=self.image.dtype)
            image_new[:y1, :]           = self.image[:y1, :]
            image_new[y1:, :]           = self.image[y2:, :]

            ocr_row = get_bounded_ocr(self.ocr, (0, y1), (self.image.shape[1], y2), remove_org=True)

            self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (0, y1), (self.image.shape[1], self.image.shape[0]), remove_org=True), (0, y1 - y2))

            self.rows = [row for row in self.rows if row not in self.rows[to_remove: to_remove + maxspan + 1]]
            self.rows[to_remove:] = [row - (y2 - y1) for row in self.rows[to_remove:]]
            self.rows.sort()
            self.image = image_new

            new_cells = [[None for i in range(len(self.columns) - 1)] for j in range(len(self.rows) - 1)]

            for i in range(len(new_cells)):
                for j in range(len(new_cells[0])):
                    if i < to_remove:
                        new_cells[i][j] = self.cells[i][j]
                    else:
                        cell = self.cells[i + maxspan + 1][j]
                        cell.attrib['y0'] -= y2 - y1
                        cell.attrib['y1'] -= y2 - y1
                        cell.attrib['startRow'] -= maxspan + 1
                        cell.attrib['endRow'] -= maxspan + 1
                        new_cells[i][j] = cell
            self.cells = new_cells
        else:
            minspan = 0
            maxspan = 0
            for i in range(len(self.cells)):
                if self.cells[i][to_remove].attrib['colspan'] != 0:
                    minspan = min(minspan, self.cells[i][to_remove].attrib['colspan'])
            minspan *= -1
            to_remove -= minspan
            for i in range(len(self.cells)):
                if self.cells[i][to_remove].attrib['colspan'] != 0:
                    maxspan = max(maxspan, self.cells[i][to_remove].attrib['colspan'])

            x1 = self.columns[to_remove]
            x2 = self.columns[to_remove + maxspan + 1]

            if x2 - x1 >= self.image.shape[1] or len(self.columns) - (maxspan + 1) < 4:
                return False

            image_col = self.image[:, x1:x2]

            image_new = np.zeros((self.image.shape[0], self.image.shape[1] - image_col.shape[1]), dtype=self.image.dtype)
            image_new[:, :x1]           = self.image[:, :x1]
            image_new[:, x1:]           = self.image[:, x2:]

            ocr_col = get_bounded_ocr(self.ocr, (x1, 0), (x2, self.image.shape[0]), remove_org=True)

            self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (x1, 0), (self.image.shape[1], self.image.shape[0]), remove_org=True), (x1 - x2, 0))

            self.columns = [col for col in self.columns if col not in self.columns[to_remove: to_remove + maxspan + 1]]
            self.columns[to_remove:] = [col - (x2 - x1) for col in self.columns[to_remove:]]
            self.columns.sort()
            self.image = image_new

            new_cells = [[None for i in range(len(self.columns) - 1)] for j in range(len(self.rows) - 1)]

            for i in range(len(new_cells)):
                for j in range(len(new_cells[0])):
                    if j < to_remove:
                        new_cells[i][j] = self.cells[i][j]
                    else:
                        cell = self.cells[i][j + maxspan + 1]
                        cell.attrib['x0'] -= x2 - x1
                        cell.attrib['x1'] -= x2 - x1
                        cell.attrib['startCol'] -= maxspan + 1
                        cell.attrib['endCol'] -= maxspan + 1
                        new_cells[i][j] = cell
            self.cells = new_cells

    def visualize(self, window="image"):
        image = self.image.copy()
        if len(self.image.shape) == 2 or self.image.shape[2] < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self.t.visualize(image)

        for word in self.ocr:
            cv2.rectangle(image, tuple(word[2:4]), tuple(word[4:6]), (0, 0, 0), 1)

        cv2.imshow(window, image)
        cv2.waitKey(0)

        # for i in range(len(self.rows)):
        #     cv2.line(image, (0, self.rows[i]), (image.shape[1], self.rows[i]), (0, 0, 255), 2)
        #     if i !=  len(self.rows) - 1:
        #         ocr = get_bounded_ocr(self.ocr, (0, self.rows[i]), (image.shape[1], self.rows[i + 1]))
        #         color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        #         for word in ocr:
        #             cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
        
        # image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        # for i in range(len(self.columns)):
        #     cv2.line(image, (self.columns[i], 0), (self.columns[i], image.shape[0]), (0, 255, 0), 2)
        #     if i !=  len(self.columns) - 1:
        #         ocr = get_bounded_ocr(self.ocr, (self.columns[i], 0), (self.columns[i + 1], image.shape[0]))
        #         color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        #         for word in ocr:
        #             cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
        # cv2.imshow("col", image)

        # image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        # for i in range(len(self.cells)):
        #     for j in range(len(self.cells[i])):
        #         if self.cells[i][j].attrib['dontCare'] == 'true':
        #             continue
        #         p1 = (self.cells[i][j].attrib['x0'], self.cells[i][j].attrib['y0'])
        #         p2 = (self.cells[i][j].attrib['x1'], self.cells[i][j].attrib['y1'])
        #         ocr = get_bounded_ocr(self.ocr, p1, p2)
        #         color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
        #         for word in ocr:
        #             cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
        #         cv2.rectangle(image, (p1[0] + 2, p1[1] + 2), (p2[0] - 4, p2[1] - 4), color, 2)
        # cv2.imshow("cell", image)
        # cv2.waitKey(0)

    # def get_xml(self):
    #     if not self.verify_cells():
    #         # print("CELLS ARE INCORRECT!")
    #         return None

    #     out_root = ET.Element("Table")
    #     out_root.attrib['x0'] = str(0)
    #     out_root.attrib['x1'] = str(self.image.shape[1])
    #     out_root.attrib['y0'] = str(0)
    #     out_root.attrib['y1'] = str(self.image.shape[0])
    #     out_root.attrib['orientation'] = self.xml_obj.attrib['orientation']

    #     for row in self.rows[1:-1]:
    #         out_row = ET.SubElement(out_root, "Row")
            
    #         out_row.attrib['x0'] = str(0)
    #         out_row.attrib['x1'] = str(self.image.shape[1])
    #         out_row.attrib['y0'] = str(row)
    #         out_row.attrib['y1'] = str(row)
            
    #     for col in self.columns[1:-1]:
    #         out_col = ET.SubElement(out_root, "Column")
            
    #         out_col.attrib['x0'] = str(col)
    #         out_col.attrib['x1'] = str(col)
    #         out_col.attrib['y0'] = str(0)
    #         out_col.attrib['y1'] = str(self.image.shape[0])

    #     for i in range(len(self.cells)):
    #         for j in range(len(self.cells[i])):
    #             for key in self.cells[i][j].attrib.keys():
    #                 self.cells[i][j].attrib[key] = str(self.cells[i][j].attrib[key])
    #             self.cells[i][j].attrib.pop('colspan')
    #             self.cells[i][j].attrib.pop('rowspan')
    #             cell = ET.SubElement(out_root, "Cell")
    #             cell.attrib = self.cells[i][j].attrib
    #             cell.text = self.cells[i][j].text

    #     return out_root


    # def verify_cells(self):
    #     try:
    #         for i in range(len(self.cells)):
    #             for j in range(len(self.cells[i])):
    #                 if self.rows[self.cells[i][j].attrib['startRow']] != self.cells[i][j].attrib['y0']:
    #                     assert False
    #                 if self.columns[self.cells[i][j].attrib['startCol']] != self.cells[i][j].attrib['x0']:
    #                     assert False
    #                 if self.rows[self.cells[i][j].attrib['endRow'] + 1] != self.cells[i][j].attrib['y1']:
    #                     assert False
    #                 if self.columns[self.cells[i][j].attrib['endCol'] + 1] != self.cells[i][j].attrib['x1']:
    #                     assert False

    #                 if self.cells[i][j].attrib['endRow'] - self.cells[i][j].attrib['startRow'] != self.cells[i][j].attrib['rowspan']:
    #                     print("Row span is not correct")
    #                     return False
    #                 if self.cells[i][j].attrib['endCol'] - self.cells[i][j].attrib['startCol'] != self.cells[i][j].attrib['colspan']:
    #                     print("Column span is not correct")
    #                     return False
    #                 if self.cells[i][j].attrib['dontCare'] == 'true':
    #                     if self.cells[i][j].attrib['rowspan'] != 0:
    #                         print("Row span of dont care cell should be 0")
    #                         return False
    #                     if self.cells[i][j].attrib['colspan'] != 0:
    #                         print("Column span of dont care cell should be 0")
    #                         return False

                    
    #                 r1 = self.cells[i][j].attrib['startRow']
    #                 c1 = self.cells[i][j].attrib['startCol']
    #                 for x in range(self.cells[i][j].attrib['colspan'] + 1):
    #                     for y in range(self.cells[i][j].attrib['rowspan'] + 1):
    #                         if x==0 and y==0:
    #                             continue
    #                         if self.cells[r1 + y][c1 + x].attrib['dontCare'] != 'true':
    #                             print("Cell should be dont care")
    #                             return False
    #                         if self.cells[r1 + y][c1 + x].attrib['colspan'] != -x:
    #                             print("Column span incorrect in merged dontcare cells")
    #                             return False
    #                         if self.cells[r1 + y][c1 + x].attrib['rowspan'] != -y:
    #                             print("Row span incorrect in merged dontcare cells")
    #                             return False
    #                         self.cells[r1 + y][c1 + x].attrib['colspan'] = 0
    #                         self.cells[r1 + y][c1 + x].attrib['rowspan'] = 0
    #     except Exception as e:
    #         print(e)
    #         return False
    #     return True
