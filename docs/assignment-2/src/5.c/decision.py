import numpy as np
from math import e
import pylab

x = np.linspace(0, 5, 1000)
pylab.plot(x, list(map(lambda x: 0.5*0.5 if 2 <= x <= 4 else 0, x)), color='darkblue')
pylab.plot(x, list(map(lambda x: 0.5*1 * e ** (-x), x)), color='orange')
pylab.fill_between(np.linspace(2, 4, 1000), list(map(lambda x: 0.5* e ** (-x), np.linspace(2, 4, 1000))),
                 color='darkgreen')
pylab.legend(['Uniform Distribution', 'Exponential Distribution', 'Bayes Error Surface'], loc='upper right') 
pylab.show()




