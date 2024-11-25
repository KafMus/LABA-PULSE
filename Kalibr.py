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

# Аппромиксация функции методом наименьших квадратов
def approx(x, y):
    a = ( np.mean( np.multiply(x, y)) - np.mean(x) * np.mean(y) ) / ( np.mean(np.pow(x, 2)) - np.mean(x)**2 )
    b = np.mean(y) - a * np.mean(x)
    return a, b

##### [###############---PRE-BEGINNING---###############]
fig, ax = plot.subplots(figsize=(18, 10), layout='constrained')

##### [###############---BEGINNING---###############]
### [#####--EXPER-DATA--#####]
Path = "D://"
data = np.array(open(Path).read().split('\n'))
data = [i.split('\t') for i in data]

pressure = [ float(i[0]) for i in data ]
num_to_conv = [ float(i[1]) for i in data ]

##### [###############---ANALYZING-DATA---###############]
ax.set_title("Kalibrovka")
ax.set_xlabel('Num to convert')
ax.set_ylabel('Pressure in MM RT. ST.')

err_x = [ 0 for i in data ]
err_y = [ 0 for i in data ]
a, b = approx(num_to_conv, pressure)
vx = np.linspace(0, max(num_to_conv), 42)

ax.plot(vx, a * vx + b)
ax.scatter(num_to_conv, pressure, c="red")


##### [###############---ENDING---###############]
print("k = ", a, "\tb = ", b)

plot.grid()
plot.savefig("D://", dpi=400)
plot.show()
