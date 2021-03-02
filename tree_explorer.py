import cv2
import copy
import random
import numpy as np

from xml.dom import minidom
from xml.etree import ElementTree as ET

from truthpy import Table, GTElement, Row, Column, RowSpan, ColSpan

class Augmentor:
    def __init__(self, table):
        self.t = table
        
        self.doc_shape = (table.y2, table.x2)
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

        if idx_c1 == 0:
            return False

        if abs(idx_p1 - idx_paste) <= abs(idx_p2 - idx_paste) and idx_p1 != 0:
            idx_paste = idx_p1
        else:
            idx_paste = idx_p2 + 1

        copy_y1 = self.t.gtCells[idx_c1][0].y1
        copy_y2 = self.t.gtCells[idx_c2][0].y2
        paste_y1 = self.t.gtCells[idx_paste - 1][0].y2
        h = copy_y2 - copy_y1

        if self.t.y2 + h > self.doc_shape[0] * 1.5:
            return False

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

        if idx_1 == 0:
            return False

        y1 = self.t.gtCells[idx_1][0].y1
        y2 = self.t.gtCells[idx_2][0].y2
        h = y2 - y1

        if h >= self.t.y2 * 0.6 or len(self.t.gtCells) - (idx_2 - idx_1 + 1) <= 3:
            return False

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

        if idx_c1 == 0:
            return False

        if abs(idx_p1 - idx_paste) <= abs(idx_p2 - idx_paste) and idx_p1 != 0:
            idx_paste = idx_p1
        else:
            idx_paste = idx_p2 + 1

        copy_x1 = self.t.gtCells[0][idx_c1].x1
        copy_x2 = self.t.gtCells[0][idx_c2].x2
        paste_x1 = self.t.gtCells[0][idx_paste - 1].x2
        w = copy_x2 - copy_x1

        if self.t.x2 + w > self.doc_shape[1] * 1.5:
            return False

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

        if idx_1 == 0:
            return False

        x1 = self.t.gtCells[0][idx_1].x1
        x2 = self.t.gtCells[0][idx_2].x2
        w = x2 - x1

        if w >= self.t.x2 * 0.7 or len(self.t.gtCells[0]) - (idx_2 - idx_1 + 1) <= 2:
            return False

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

    def visualize(self, window="image"):
        image = np.ones((self.t.y2, self.t.x2, 3), dtype=np.uint8) * 255
        if len(image.shape) == 2 or image.shape[2] < 3:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        self.t.visualize(image)

        cv2.imshow(window, image)
        cv2.waitKey(0)

# MAX_DEPTH = 5
# MAX_WIDTH = [10, 5, 2, 1, 1]
# MAX_ATTEMPTS = [10, 5, 2, 1, 1]

MAX_DEPTH = 10
MAX_WIDTH = [8, 4, 2, 2, 2, 1, 1, 1, 1, 1]
MAX_ATTEMPTS = [x * 2 for x in MAX_WIDTH]

def apply_action(obj, action):
    return getattr(obj, action[0])(*action[1])

def DFS(augmentor, depth, actions):

    if augmentor.t.y2 > augmentor.doc_shape[0] * 1.5 or augmentor.t.x2 > augmentor.doc_shape[1] * 1.5 or depth < 5:
        return_list = []
    else:
        return_list = [(
            augmentor.t, 
            {
                "h" : len(augmentor.t.gtCells),
                "w" : len(augmentor.t.gtCells[0]),
                "col_spans" : len([x for x in augmentor.t.gtSpans if isinstance(x, ColSpan)]),
                "row_spans" : len([x for x in augmentor.t.gtSpans if isinstance(x, RowSpan)]),
            },
            actions
        )]

    if depth < MAX_DEPTH:
        counter = 0
        for i in range(MAX_ATTEMPTS[depth]):
            if counter >= MAX_WIDTH[depth]:
                break

            prob = random.random()

            tmp = copy.deepcopy(augmentor)
            if prob < 0.25:
                size = len(tmp.t.gtCells[0])
                i = random.choice(range(1, size))
                j = random.choice(range(1, size + 1))
                action = ('replicate_column', (i, j))

            elif prob < 0.5:
                size = len(tmp.t.gtCells[0])
                i = random.choice(range(1, size))
                action = ('remove_column', (i,))

            elif prob < 0.75:
                size = len(tmp.t.gtCells)
                i = random.choice(range(1, size))
                j = random.choice(range(1, size + 1))
                action = ('replicate_row', (i, j))

            else:
                size = len(tmp.t.gtCells)
                i = random.choice(range(1, size))
                action = ('remove_row', (i,))

            edited = apply_action(tmp, action)
            if edited:
                counter += 1
                actions_copy = actions.copy()
                actions_copy.append(action)
                return_list += DFS(tmp, depth + 1, actions_copy)
    return return_list

def dfs_wrapper(table):
    return DFS(Augmentor(table), 0, [])