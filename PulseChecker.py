from logging import fatal
from math import factorial

import pandas as pd
from fontTools.t1Lib import write
from scipy.optimize import curve_fit
import matplotlib.pyplot as plot
import numpy as np
import random as rnd
import math

from scipy.stats import expon

def factor(x):
    res = 1
    for i in range(2, x + 1):
        res *= i
    return res

# Аппромиксация функции методом наименьших квадратов
def approx(x, y):
    a = ( np.mean( np.multiply(x, y)) - np.mean(x) * np.mean(y) ) / ( np.mean(np.pow(x, 2)) - np.mean(x)**2 )
    b = np.mean(y) - a * np.mean(x)
    return a, b

# АНТИ-ШУМ
def anti_noise(x, y, step):
    parts_x = []
    last_min_x = min(x)
    while True:
        if (last_min_x > max(x)):
            break
        parts_x.append([ i for i in x if i >= last_min_x and i < last_min_x + step ])
        if len(parts_x[len(parts_x) - 1]) == 0:
            parts_x.pop(len(parts_x) - 1)
            parts_x.append(parts_x[len(parts_x) - 1])
        last_min_x += step
    parts_indexes = [[ x.index(j) for j in i ] for i in parts_x]
    parts_y = [ [ y[j] for j in i ] for i in parts_indexes ]
    new_y = [ np.mean(i) for i in parts_y]
    new_x = [ np.mean(i) for i in parts_x]
    return new_x, new_y

##### [###############---PRE-BEGINNING---###############]
fig, ax = plot.subplots(figsize=(18, 10), layout='constrained')

#plot.xlim(left=9.08)
#plot.xlim(right=9.8)
#plot.ylim(top=0.0097)
#plot.ylim(bottom=0.0085)

##### [###############---BEGINNING---###############]
### [#####--INIT-DATA--#####]
k = 0.10333220704574829
b = 0

### [#####--EXPER-DATA--#####]
Path = "D://"
data = np.array(open(Path).read().split('\n'))
data = [i.split(' ') for i in data]

num_to_conv = [ float(i[0]) for i in data ]
time = [ float(i[1]) for i in data ]

### [#####--RESULT-DATA--#####]
pressure = [ (k * i + b) for i in num_to_conv ]

ax.set_title("Pulse check")
ax.set_xlabel('Time')
ax.set_ylabel('Pressure in MM RT. ST.')

# ax.scatter(time, pressure, s=1)


# #### [###############---ANALYZING-DATA---###############]
# ## [#####---CUTTING---#####]
left_edge = 6.5
right_edge = 13

new_time     = [ i for i in time if i >= left_edge and i <= right_edge ]
new_indexes  = [ time.index(i) for i in new_time ]
new_pressure = [ pressure[i] for i in new_indexes ]
# ax.scatter(new_time, new_pressure, s=1)

###################################
### [#####---PARTED-APPROX---#####]

step_appr = 1.5
parts_time = []
parts_pressure = []
min_parts_time = min(new_time)
for j in range(0, int(len(new_time) / step_appr)):
    parts_time.append([ i for i in new_time if i >= min_parts_time and i < min_parts_time + step_appr ])
    parts_pressure.append([ new_pressure[i] for i in range(0, len(new_pressure)) if new_time[i] >= min_parts_time and new_time[i] < min_parts_time + step_appr ])
    min_parts_time += step_appr
a_appr = []
b_appr = []
for j in range(0, len(parts_time)):
    tmp_a, tmp_b = approx(parts_time[j], parts_pressure[j])
    a_appr.append(tmp_a)
    b_appr.append(tmp_b)

### [#####---PARTED-NORMIS---#####]

new_parts_pressure = [ [( parts_pressure[j][i] - (a_appr[j] * parts_time[j][i] + b_appr[j]) ) for i in range(0, len(parts_time[j]))] for j in range(0, len(parts_time)) ]

new_pressure = []
new_time = []
for j in new_parts_pressure:
    for i in j:
        new_pressure.append(i)
for j in parts_time:
    for i in j:
        new_time.append(i)
# ax.scatter(new_time, new_pressure, s=1)


### [#####---NOISE-CANCELLING---#####]
new_time, new_pressure = anti_noise(new_time, new_pressure, 0.0002)
# # ax.scatter(new_time, new_pressure, s=(4200 / len(new_time)) )
ax.plot(new_time, new_pressure )

##### [###############---ENDING---###############]

plot.grid()
# plot.savefig("D://", dpi=400)
plot.show()
