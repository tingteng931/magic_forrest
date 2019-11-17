# -*- coding: utf-8 -*-

import cv2
import numpy as np

img = cv2.imread('img/spells-01.png')
spells = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
test_img = cv2.imread('img/test_spell_3.png')
test_spells = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

rows = np.vsplit(spells, 10)

cells = []
for row in rows:
	row_cells = np.hsplit(row, 10)
	for cell in row_cells:
		cell=cell.flatten()
		cells.append(cell)
cells = np.array(cells, dtype=np.float32)

k = np.arange(5)
cells_labels = np.repeat(k, 20)

test_spells = np.vsplit(test_spells, 1)
test_cells = []
for d in test_spells:
	d = d.flatten()
	test_cells.append(d)

test_cells = np.array(test_cells, dtype=np.float32)

knn = cv2.ml.KNearest_create()
knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
ret, result, neighbours, dist = knn.findNearest(test_cells, k=1)

print("the result is", int(result[0][0]))


