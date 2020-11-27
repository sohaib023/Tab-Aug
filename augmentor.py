import cv2
import copy
import random
import numpy as np

from xml.dom import minidom
from xml.etree import ElementTree as ET

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

class Rect:
    """internal custom class used to perform rectangular calculations used for evaluation"""

    def __init__(self, x1, y1, x2, y2, prob=0.0, text=None):
        self.prob = prob
        self.text = text
        self.init(x1, y1, x2, y2)

    def init_bbox(self, x1, y1, x2, y2):
        w = x2 - x1
        h = y2 - y1
        if x1 > x2 or y1 > y2:
            raise ValueError("Coordinates are invalid")
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.w, self.h = w, h

    def area_diff(self, other):
        """calculates the area of box that is non-intersecting with 'other' rect"""
        return self.area() - self.intersection(other).area()

    def area(self):
        """calculates the area of the box"""
        return self.w * self.h

    def intersection(self, other):
        """calculates the intersecting area of two rectangles"""
        a, b = self, other
        x1 = max(a.x1, b.x1)
        y1 = max(a.y1, b.y1)
        x2 = min(a.x2, b.x2)
        y2 = min(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2 - x1, y2 - y1)
        else:
            return type(self)(0, 0, 0, 0)

    __and__ = intersection

    def union(self, other):
        """takes the union of the two rectangles"""
        a, b = self, other
        x1 = min(a.x1, b.x1)
        y1 = min(a.y1, b.y1)
        x2 = max(a.x2, b.x2)
        y2 = max(a.y2, b.y2)
        if x1 < x2 and y1 < y2:
            return type(self)(x1, y1, x2 - x1, y2 - y1)

    __or__ = union
    __sub__ = area_diff

    def __iter__(self):
        yield self.x1
        yield self.y1
        yield self.x2
        yield self.y2

    def __eq__(self, other):
        return isinstance(other, Rect) and tuple(self) == tuple(other)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return type(self).__name__ + repr(tuple(self))


class Table(Rect):
    def __init__(self, image, xml_obj, ocr):
        self.xml_obj = xml_obj

        columns = []
        rows = []
        self.init_bbox(
            int(xml_obj.attrib["x0"]),
            int(xml_obj.attrib["y0"]),
            int(xml_obj.attrib["x1"]),
            int(xml_obj.attrib["y1"])
        )
        
        self.image_limit = image.shape[:2]
        self.image = image[self.y1:self.y2, self.x1:self.x2]

        for col in xml_obj.findall(".//Column"):
            columns.append(int(col.attrib['x0']) - self.x1)
        for row in xml_obj.findall(".//Row"):
            rows.append(int(row.attrib['y0']) - self.y1)

        self.rows = [0] + rows + [self.h]
        self.columns = [0] + columns + [self.w]
        self.rows.sort()
        self.columns.sort()

        self.cells = [[None for i in range(len(self.columns) - 1)] for j in range(len(self.rows) - 1)]

        self.ocr = translate_ocr(get_bounded_ocr(ocr, (self.x1, self.y1), (self.x2, self.y2)), (-self.x1, -self.y1))

        for cell in xml_obj.findall(".//Cell"):
            for key in cell.attrib.keys():
                if key == 'dontCare':
                    continue
                cell.attrib[key] = int(cell.attrib[key])
            cell.attrib['colspan'] = (cell.attrib['endCol'] - cell.attrib['startCol'])
            cell.attrib['rowspan'] = (cell.attrib['endRow'] - cell.attrib['startRow'])
            cell.attrib['x0'] -= self.x1
            cell.attrib['y0'] -= self.y1
            cell.attrib['x1'] -= self.x1
            cell.attrib['y1'] -= self.y1

            self.cells[cell.attrib['startRow']][cell.attrib['startCol']] = cell

        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                r1 = self.cells[i][j].attrib['startRow']
                c1 = self.cells[i][j].attrib['startCol']
                for x in range(self.cells[i][j].attrib['colspan'] + 1):
                    for y in range(self.cells[i][j].attrib['rowspan'] + 1):
                        if x==0 and y==0:
                            continue
                        self.cells[r1 + y][c1 + x].attrib['colspan'] = -x
                        self.cells[r1 + y][c1 + x].attrib['rowspan'] = -y

    def replicate(self, idx_copy, idx_paste, is_row):
        if is_row:
            minspan = 0
            maxspan = 0
            for i in range(len(self.cells[idx_copy])):
                if self.cells[idx_copy][i].attrib['rowspan'] != 0:
                    minspan = min(minspan, self.cells[idx_copy][i].attrib['rowspan'])
            minspan *= -1
            idx_copy -= minspan
            for i in range(len(self.cells[idx_copy])):
                if self.cells[idx_copy][i].attrib['rowspan'] != 0:
                    maxspan = max(maxspan, self.cells[idx_copy][i].attrib['rowspan'])
            if idx_paste < len(self.cells):
                minspan2 = 0
                maxspan2 = 0
                for i in range(len(self.cells[idx_paste])):
                    if self.cells[idx_paste][i].attrib['rowspan'] != 0:
                        minspan2 = min(minspan2, self.cells[idx_paste][i].attrib['rowspan'])
                        maxspan2 = max(maxspan2, self.cells[idx_paste][i].attrib['rowspan'])
                idx_paste = idx_paste + minspan2

            copy_y1 = self.rows[idx_copy]
            h = self.rows[idx_copy + maxspan + 1] - copy_y1
            paste_y1 = self.rows[idx_paste]

            image_row = self.image[copy_y1:copy_y1 + h, :]

            image_new = np.zeros((self.image.shape[0] + image_row.shape[0], self.image.shape[1]), dtype=self.image.dtype)
            image_new[:paste_y1, :]             = self.image[:paste_y1, :]
            image_new[paste_y1:paste_y1 + h, :] = image_row
            image_new[paste_y1 + h:, :]         = self.image[paste_y1:, :]

            ocr_row = get_bounded_ocr(self.ocr, (0, copy_y1), (self.image.shape[1], copy_y1 + h))
            ocr_row = translate_ocr(ocr_row, (0, paste_y1 - copy_y1))

            self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (0, paste_y1), (self.image.shape[1], self.image.shape[0]), remove_org=True), (0, h))
            self.ocr += ocr_row

            new_rows = [(row - copy_y1) + paste_y1 for row in self.rows[idx_copy: idx_copy + maxspan + 1]]
            self.rows[idx_paste:] = [row + h for row in self.rows[idx_paste:]]
            self.rows += new_rows
            self.rows.sort()
            self.image = image_new

            new_cells = [[None for i in range(len(self.columns) - 1)] for j in range(len(self.rows) - 1)]

            for i in range(len(new_cells)):
                for j in range(len(new_cells[0])):
                    if i < idx_paste:
                        new_cells[i][j] = self.cells[i][j]
                    elif i < idx_paste + len(new_rows):
                        cell = copy.deepcopy(self.cells[(i - idx_paste) + idx_copy][j])
                        cell.attrib['y0'] += paste_y1 - copy_y1
                        cell.attrib['y1'] += paste_y1 - copy_y1
                        cell.attrib['startRow'] += idx_paste - idx_copy
                        cell.attrib['endRow'] += idx_paste - idx_copy
                        new_cells[i][j] = cell
                    else:
                        cell = self.cells[i - len(new_rows)][j]
                        cell.attrib['y0'] += h
                        cell.attrib['y1'] += h
                        cell.attrib['startRow'] += len(new_rows)
                        cell.attrib['endRow'] += len(new_rows)
                        new_cells[i][j] = cell
            self.cells = new_cells

        else:
            minspan = 0
            maxspan = 0
            for i in range(len(self.cells)):
                if self.cells[i][idx_copy].attrib['colspan'] != 0:
                    minspan = min(minspan, self.cells[i][idx_copy].attrib['colspan'])
            minspan *= -1
            idx_copy -= minspan
            for i in range(len(self.cells)):
                if self.cells[i][idx_copy].attrib['colspan'] != 0:
                    maxspan = max(maxspan, self.cells[i][idx_copy].attrib['colspan'])
            if idx_paste < len(self.cells[0]):
                minspan2 = 0
                for i in range(len(self.cells)):
                    if self.cells[i][idx_paste].attrib['colspan'] != 0:
                        minspan2 = min(minspan2, self.cells[i][idx_paste].attrib['colspan'])
                idx_paste = idx_paste + minspan2

            copy_x1 = self.columns[idx_copy]
            w = self.columns[idx_copy + maxspan + 1] - copy_x1
            paste_x1 = self.columns[idx_paste]

            if self.image.shape[1] + w > self.image_limit[1] *0.9:
                return False

            image_col = self.image[:, copy_x1:copy_x1 + w]

            image_new = np.zeros((self.image.shape[0], self.image.shape[1] + image_col.shape[1]), dtype=self.image.dtype)
            image_new[:, :paste_x1]             = self.image[:, :paste_x1]
            image_new[:, paste_x1: paste_x1+w]  = image_col
            image_new[:, paste_x1+w:]           = self.image[:, paste_x1:]

            ocr_col = get_bounded_ocr(self.ocr, (copy_x1, 0), (copy_x1 + w, self.image.shape[0]))
            ocr_col = translate_ocr(ocr_col, (paste_x1 - copy_x1, 0))

            self.ocr += translate_ocr(get_bounded_ocr(self.ocr, (paste_x1, 0), (self.image.shape[1], self.image.shape[0]), remove_org=True), (w, 0))
            self.ocr += ocr_col

            new_columns = [(col - copy_x1) + paste_x1 for col in self.columns[idx_copy: idx_copy + maxspan + 1]]
            self.columns[idx_paste:] = [col + w for col in self.columns[idx_paste:]]
            self.columns += new_columns
            self.columns.sort()
            self.image = image_new

            # for i in range(len(self.cells[idx_copy])):
            #     cells_row.append(copy.deepcopy(self.cells[idx_copy][i]))

            new_cells = [[None for i in range(len(self.columns) - 1)] for j in range(len(self.rows) - 1)]

            for i in range(len(new_cells)):
                for j in range(len(new_cells[0])):
                    if j < idx_paste:
                        new_cells[i][j] = self.cells[i][j]
                    elif j < idx_paste + len(new_columns):
                        cell = copy.deepcopy(self.cells[i][(j - idx_paste) + idx_copy])
                        cell.attrib['x0'] += paste_x1 - copy_x1
                        cell.attrib['x1'] += paste_x1 - copy_x1
                        cell.attrib['startCol'] += idx_paste - idx_copy
                        cell.attrib['endCol'] += idx_paste - idx_copy
                        new_cells[i][j] = cell
                    else:
                        cell = self.cells[i][j - len(new_columns)]
                        cell.attrib['x0'] += w
                        cell.attrib['x1'] += w
                        cell.attrib['startCol'] += len(new_columns)
                        cell.attrib['endCol'] += len(new_columns)
                        new_cells[i][j] = cell

            self.cells = new_cells            

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

    def visualize(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)

        # print(len(self.rows))
        for i in range(len(self.rows)):
            cv2.line(image, (0, self.rows[i]), (image.shape[1], self.rows[i]), (0, 0, 255), 2)
            if i !=  len(self.rows) - 1:
                ocr = get_bounded_ocr(self.ocr, (0, self.rows[i]), (image.shape[1], self.rows[i + 1]))
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                for word in ocr:
                    cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
        cv2.imshow("row", image)
        
        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for i in range(len(self.columns)):
            cv2.line(image, (self.columns[i], 0), (self.columns[i], image.shape[0]), (0, 255, 0), 2)
            if i !=  len(self.columns) - 1:
                ocr = get_bounded_ocr(self.ocr, (self.columns[i], 0), (self.columns[i + 1], image.shape[0]))
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                for word in ocr:
                    cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
        cv2.imshow("col", image)

        image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                if self.cells[i][j].attrib['dontCare'] == 'true':
                    continue
                p1 = (self.cells[i][j].attrib['x0'], self.cells[i][j].attrib['y0'])
                p2 = (self.cells[i][j].attrib['x1'], self.cells[i][j].attrib['y1'])
                ocr = get_bounded_ocr(self.ocr, p1, p2)
                color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
                for word in ocr:
                    cv2.rectangle(image, (word[2], word[3]), (word[4], word[5]), color, 2)
                cv2.rectangle(image, (p1[0] + 2, p1[1] + 2), (p2[0] - 4, p2[1] - 4), color, 2)
        cv2.imshow("cell", image)
        cv2.waitKey(0)

    def get_xml(self):
        if not self.verify_cells():
            # print("CELLS ARE INCORRECT!")
            return None

        out_root = ET.Element("Table")
        out_root.attrib['x0'] = str(0)
        out_root.attrib['x1'] = str(self.image.shape[1])
        out_root.attrib['y0'] = str(0)
        out_root.attrib['y1'] = str(self.image.shape[0])
        out_root.attrib['orientation'] = self.xml_obj.attrib['orientation']

        for row in self.rows[1:-1]:
            out_row = ET.SubElement(out_root, "Row")
            
            out_row.attrib['x0'] = str(0)
            out_row.attrib['x1'] = str(self.image.shape[1])
            out_row.attrib['y0'] = str(row)
            out_row.attrib['y1'] = str(row)
            
        for col in self.columns[1:-1]:
            out_col = ET.SubElement(out_root, "Column")
            
            out_col.attrib['x0'] = str(col)
            out_col.attrib['x1'] = str(col)
            out_col.attrib['y0'] = str(0)
            out_col.attrib['y1'] = str(self.image.shape[0])

        for i in range(len(self.cells)):
            for j in range(len(self.cells[i])):
                for key in self.cells[i][j].attrib.keys():
                    self.cells[i][j].attrib[key] = str(self.cells[i][j].attrib[key])
                self.cells[i][j].attrib.pop('colspan')
                self.cells[i][j].attrib.pop('rowspan')
                cell = ET.SubElement(out_root, "Cell")
                cell.attrib = self.cells[i][j].attrib
                cell.text = self.cells[i][j].text

        return out_root


    def verify_cells(self):
        try:
            for i in range(len(self.cells)):
                for j in range(len(self.cells[i])):
                    if self.rows[self.cells[i][j].attrib['startRow']] != self.cells[i][j].attrib['y0']:
                        assert False
                    if self.columns[self.cells[i][j].attrib['startCol']] != self.cells[i][j].attrib['x0']:
                        assert False
                    if self.rows[self.cells[i][j].attrib['endRow'] + 1] != self.cells[i][j].attrib['y1']:
                        assert False
                    if self.columns[self.cells[i][j].attrib['endCol'] + 1] != self.cells[i][j].attrib['x1']:
                        assert False

                    if self.cells[i][j].attrib['endRow'] - self.cells[i][j].attrib['startRow'] != self.cells[i][j].attrib['rowspan']:
                        print("Row span is not correct")
                        return False
                    if self.cells[i][j].attrib['endCol'] - self.cells[i][j].attrib['startCol'] != self.cells[i][j].attrib['colspan']:
                        print("Column span is not correct")
                        return False
                    if self.cells[i][j].attrib['dontCare'] == 'true':
                        if self.cells[i][j].attrib['rowspan'] != 0:
                            print("Row span of dont care cell should be 0")
                            return False
                        if self.cells[i][j].attrib['colspan'] != 0:
                            print("Column span of dont care cell should be 0")
                            return False

                    
                    r1 = self.cells[i][j].attrib['startRow']
                    c1 = self.cells[i][j].attrib['startCol']
                    for x in range(self.cells[i][j].attrib['colspan'] + 1):
                        for y in range(self.cells[i][j].attrib['rowspan'] + 1):
                            if x==0 and y==0:
                                continue
                            if self.cells[r1 + y][c1 + x].attrib['dontCare'] != 'true':
                                print("Cell should be dont care")
                                return False
                            if self.cells[r1 + y][c1 + x].attrib['colspan'] != -x:
                                print("Column span incorrect in merged dontcare cells")
                                return False
                            if self.cells[r1 + y][c1 + x].attrib['rowspan'] != -y:
                                print("Row span incorrect in merged dontcare cells")
                                return False
                            self.cells[r1 + y][c1 + x].attrib['colspan'] = 0
                            self.cells[r1 + y][c1 + x].attrib['rowspan'] = 0
        except Exception as e:
            print(e)
            return False
        return True
