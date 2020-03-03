"""
@project = part_time
@file = test
@author = 10374
@create_time = 2019/10/27 16:13
"""

import os
import math
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

num = 9
l = 3
Action = []
for i in range(num):
	tmp_a = np.ones(l, dtype=np.int64)
	j = l - 1
	tmp_e = i
	if tmp_e == 1:
		units = i % 4
		tmp_a[j] += units
		tens = i / 4
		tmp_a[j - 1] += tens
	while tmp_e != 1:
		units = tmp_e % 4
		tmp_a[j] += units
		tens = tmp_e / 4
		if tens < 4:
			tmp_a[j - 1] += tens
			break
		else:
			tmp_e /= 4
			tmp_e = math.floor(tmp_e)
			j -= 1
	Action.insert(i, tmp_a)

print(Action)