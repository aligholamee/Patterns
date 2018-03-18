import numpy as np
import seaborn as sns
import pylab


data_uniform = np.random.normal(2, 4, 400)
data_exponentioal = np.random.exponential(1, 400)
x = np.arange(-5, 15, 2)
y = x - 0.693
sns.set_style('darkgrid')
sns.kdeplot(0.5*data_uniform, label='P(c1)p(x|c1)')
sns.kdeplot(0.5*data_exponentioal, label='P(c2)p(x|c2)')
pylab.plot(x, y)
pylab.axis([-1, 4, 0, 0.3])
pylab.show()
