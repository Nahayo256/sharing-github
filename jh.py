#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', '')
x = randn(10000)
hist(x, 100)


# In[ ]:





# In[3]:


import matplotlib.pyplot as plt
plt.plot([1, 3, 2, 4])

plt.show()


# In[ ]:





# In[ ]:





# In[3]:


import matplotlib.pyplot as plt
x = range(6)
plt.plot(x, [xi**2 for xi in x])
plt.show()


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0.0, 6.0, 0.01)
plt.plot(x, [x**2 for x in x])

plt.show()


# In[9]:


import matplotlib.pyplot as plt
x = range(1, 5)
plt.plot(x, [xi*1.5 for xi in x])
plt.plot(x, [xi*3.0 for xi in x])
plt.plot(x, [xi/3.0 for xi in x])

plt.show()


# In[2]:


import matplotlib.pyplot as plt
x = range(1, 5)
plt.plot(x, [xi*1.5 for xi in x], x, [xi*3.0 for xi in x], x,[xi/3.0 for xi in x])

plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)

plt.show()


# In[7]:


import matplotlib.pyplot as plt
plt.interactive(True)

plt.plot([1, 2, 3])
plt.plot([2, 4, 6])


# In[13]:


import numpy as np
x = np.array([1, 2, 3])
x
x[1:]
x[2]
x*2
l = [1, 2, 3]
[2*li for li in l]
a = np.array([1, 2, 3])
b = np.array([3, 2, 1])
a+b
M = np.array([[1, 2, 3], [4, 5, 6]])
M[1,2]
range(6)
np.arange(6)





# In[16]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.grid(True)
plt.show()


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.axis()
plt.axis([0, 5, -1, 13])
plt.show()


# In[18]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, x, x*3.0, x, x/3.0)
plt.xlim()
plt.ylim()


# In[19]:


import matplotlib.pyplot as plt
plt.plot([1, 3, 2, 4])
plt.xlabel('This is the X axis')
plt.ylabel('This is the Y axis')
plt.show()


# In[20]:


import matplotlib.pyplot as plt
plt.plot([1, 3, 2, 4])
plt.title('Simple plot')
plt.show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, label='Normal')
plt.plot(x, x*3.0, label='Fast')
plt.plot(x, x/3.0, label='Slow')
plt.legend()
plt.show()


# In[22]:


plt.plot(x, x*1.5)
plt.plot(x, x*3.0)
plt.plot(x, x/3.0)
plt.legend(['Normal', 'Fast', 'Slow'])


# In[23]:


plt.plot(x, x*1.5)
plt.plot(x, x*3.0)
plt.plot(x, x/3.0)
plt.legend(['Normal', 'Fast', 'Slow'])


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(1, 5)
plt.plot(x, x*1.5, label='Normal')
plt.plot(x, x*3.0, label='Fast')
plt.plot(x, x/3.0, label='Slow')
plt.grid(True)
plt.title('Sample Growth of a Measure')
plt.xlabel('Samples')
plt.ylabel('Values Measured')
plt.legend(loc='upper left')
plt.show()


# In[25]:


import matplotlib.pyplot as plt
plt.plot([1, 2, 3])
plt.savefig('plot123.png')


# In[26]:


import matplotlib as mpl
mpl.rcParams['figure.figsize']
mpl.rcParams['savefig.dpi']


# In[27]:


import matplotlib as mpl
mpl.rcParams['figure.figsize']
mpl.rcParams['savefig.dpi']
plt.savefig('plot123_2.png', dpi=200)


# In[28]:


import matplotlib as mpl
mpl.use('Agg') #before importing pyplot
import matplotlib.pyplot as plt
plt.plot([1,2,3])
plt.savefig('plot123_3.png')


# In[8]:


import matplotlib as mpl
import matplotlib.pyplot as plt
plt.plot([1, 3, 2, 4])

plt.show()


# In[33]:


import matplotlib as mpl
mpl.rcParams['interactive']
mpl.interactive(False)
mpl.rcParams['interactive']


# In[34]:


import matplotlib.pyplot as plt
plt.plot([1, 2])
plt.plot([2, 1]);


# In[2]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3)
plt.plot(y, 'y');
plt.plot(y+1, 'm');
plt.plot(y+2, 'c');

plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3)
plt.plot(y, 'y', y+1, 'm', y+2, 'c');

plt.show()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3)
plt.plot(y, '--', y+1, '-.', y+2, ':');

plt.show()


# In[7]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3, 0.2)
plt.plot(y, 'x', y+0.5, 'o', y+1, 'D', y+1.5, '^', y+2, 's');
plt.show()


# In[8]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3, 0.3)
plt.plot(y, 'cx--', y+1, 'mo:', y+2, 'kp-.');
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import numpy as np
y = np.arange(1, 3, 0.3)
plt.plot(y, color='blue', linestyle='dashdot', linewidth=4,
marker='o', markerfacecolor='red', markeredgecolor='black',
markeredgewidth=3, markersize=12);
plt.show()


# In[20]:


import matplotlib.pyplot as plt
x = [5, 3, 7, 2, 4, 1]
plt.plot(x);
plt.xticks(range(len(x)), ['a', 'b', 'c', 'd', 'e', 'f']);
plt.yticks(range(1, 8, 2));
plt.show()


# In[26]:


import matplotlib.pyplot as plt
import numpy as np
y = np.random.randn(1000)
plt.hist(y);
plt.show()
plt.hist(y, 25);
plt.show()


# In[31]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 4, 0.2)
y = np.exp(-x)
e1 = 0.1 * np.abs(np.random.randn(len(y)))
plt.errorbar(x, y, yerr=e1, fmt='.-');
plt.show()
e2 = 0.1 * np.abs(np.random.randn(len(y)))
plt.errorbar(x, y, yerr=e1, xerr=e2, fmt='.-', capsize=0);
plt.show()
plt.errorbar(x, y, yerr=[e1, e2], fmt='.-');
plt.show()


# In[32]:


import matplotlib.pyplot as plt
plt.bar([1, 2, 3], [3, 2, 5]);
plt.show()


# In[6]:


import matplotlib.pyplot as plt
import numpy as np
dict = {'A': 40, 'B': 70, 'C': 30, 'D': 85}
for i, key in enumerate(dict): plt.bar(i, dict[key]);
plt.xticks(np.arange(len(dict))+0.4, dict.keys());
plt.yticks(list(dict.values()));
a = dict.values ()
print(type(a))
plt.show()


# In[37]:


import matplotlib.pyplot as plt
import numpy as np
data1 = 10*np.random.rand(5)
data2 = 10*np.random.rand(5)
data3 = 10*np.random.rand(5)
e2 = 0.5 * np.abs(np.random.randn(len(data2)))
locs = np.arange(1, len(data1)+1)
width = 0.27
plt.bar(locs, data1, width=width);
plt.bar(locs+width, data2, yerr=e2, width=width,
color='red');
plt.bar(locs+2*width, data3, width=width, color='green') ;
plt.xticks(locs + width*1.5, locs);
plt.show()


# In[44]:


import matplotlib.pyplot as plt
plt.figure(figsize=(3,3));
x = [45, 35, 20]
labels = ['Cats', 'Dogs', 'Fishes']
plt.pie(x, labels=labels);
plt.show()


# In[45]:


import matplotlib.pyplot as plt
plt.figure(figsize=(3,3));
x = [4, 9, 21, 55, 30, 18]
labels = ['Swiss', 'Austria', 'Spain', 'Italy', 'France',
'Benelux']
explode = [0.2, 0.1, 0, 0, 0.1, 0]
plt.pie(x, labels=labels, explode=explode, autopct='%1.1f%%');
plt.show()


# In[50]:


import matplotlib.pyplot as plt
import numpy as np
x = np.random.randn(1000)
y = np.random.randn(1000)
plt.scatter(x, y);
plt.show()
size = 50*np.random.randn(1000)
colors = np.random.rand(1000)
plt.scatter(x, y, s=size, c=colors);
plt.show()


# In[51]:


import matplotlib.pyplot as plt
import numpy as np
theta = np.arange(0., 2., 1./180.)*np.pi
plt.polar(3*theta, theta/5);
plt.polar(theta, np.cos(4*theta));
plt.polar(theta, [1.4]*len(theta));
plt.show()


# In[54]:


import matplotlib.pyplot as plt
import numpy as np
theta = np.arange(0., 2., 1./180.)*np.pi
r = np.abs(np.sin(5*theta) - 2.*np.cos(theta))
plt.polar(theta, r);
plt.thetagrids(range(25, 360, 90));
plt.rgrids(np.arange(0.2, 3.1, .7), angle=0);
plt.show()


# In[58]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 2*np.pi, .01)
y = np.sin(x)
plt.plot(x, y);
plt.text(0.1, -0.04, 'sin(0)=0');
plt.show()


# In[61]:


import matplotlib.pyplot as plt
y = [13, 11, 13, 12, 13, 10, 30, 12, 11, 13, 12, 12, 12, 11,12]
plt.plot(y);
plt.ylim(ymax=35);
plt.annotate('this spot must really\nmean something',xy=(6, 30), xytext=(8, 31.5), arrowprops=dict(facecolor='black',shrink=0.05));
plt.show()


# In[10]:


import matplotlib.pyplot as plt
plt.axis([0, 10, 0, 20]);
arrstyles = ['-', '->', '-[', '<-', '<->', 'fancy', 'simple','wedge']
for i, style in enumerate(arrstyles):
    plt.annotate(style, xytext=(1, 2+2*i), xy=(4, 1+2*i),arrowprops=dict(arrowstyle=style));
    connstyles=["arc", "arc,angleA=10,armA=30,rad=15","arc3,rad=.2", "arc3,rad=-.2", "angle", "angle3"]
for i, style in enumerate(connstyles):
    plt.annotate("", xytext=(6, 2+2*i), xy=(8, 1+2*i),
    arrowprops=dict(arrowstyle='->', connectionstyle=style));
plt.show()


# In[ ]:





# In[9]:


plt.plot(x, np.sin(x - 0), color='blue')
plt.plot(x, np.sin(x - 1), color='g')       
plt.plot(x, np.sin(x - 2), color='0.75')        
plt.plot(x, np.sin(x - 3), color='#FFDD44') 
plt.plot(x, np.sin(x - 4), color=(1.0,0.2,0.3))
plt.plot(x, np.sin(x - 5), color='chartreuse');


# In[14]:


rng = np.random.RandomState(0)
x = rng.randn(10y = rng.randn(100))
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
plt.scatter(x, y, c=colors, s=sizes, alpha=0.3, cmap='viridis')plt.colorbar();


# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import numpy as np
def f(x, y):
    return np.sin(x) ** 10 + np.cos(10 + y * x) * np.cos(x)
    x = np.linspace(0, 5, 50)
    y = np.linspace(0, 5, 40)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
        


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.1)
y = np.random.randn(len(x))
plt.plot(x, y)
plt.title('random numbers')
plt.show()


# In[18]:


import matplotlib.pyplot as plt
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot([1, 2, 3], [1, 2, 3]);
ax2 = fig.add_subplot(212)
ax2.plot([1, 2, 3], [3, 2, 1]);
plt.show()


# In[31]:


x = arange(0, 10, 0.1)
y = randn(len(x))
plot(x, y)
title('2, 5, 10, 25, 8')
show()


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0, 10, 0.1)
y = np.random.randn(len(x))
fig = plt.figure()
ax = fig.add_subplot(111)
l, = plt.plot(x, y)
t = ax.set_title('random numbers')
plt.show()


# In[22]:


import matplotlib.pyplot as plt
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.plot([1, 2, 3], [1, 2, 3]);
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.plot([1, 2, 3], [3, 2, 1]);
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1);
ax1.set_ylabel('Y values for exp(-x)');


# In[24]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0., np.e, 0.01)
y1 = np.exp(-x)
y2 = np.log(x)
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(x, y1);
ax1.set_ylabel('Y values for exp(-x)');
ax2 = ax1.twinx() 
ax2.plot(x, y2, 'r');
ax2.set_xlim([0, np.e]);
ax2.set_ylabel('Y values for ln(x)');
ax2.set_xlabel('Same X for both exp(-x) and ln(x)');
plt.show()


# In[30]:


import matplotlib as mpl
mpl.rcParams['font.size'] = 10.
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(0., 20, 0.01)
fig = plt.figure()

ax1 = fig.add_subplot(311)
y1 = np.exp(x/6.)
ax1.plot(x, y1);
ax1.grid(True)
ax1.set_yscale('log')
ax1.set_ylabel('log Y');

ax2 = fig.add_subplot(312)
y2 = np.cos(np.pi*x)
ax2.semilogx(x, y2);
ax2.set_xlim([0, 20]);
ax2.grid(True)
ax2.set_ylabel('log X');

ax3 = fig.add_subplot(313)
y3 = np.exp(x/4.)
ax3.loglog(x, y3, basex=3);
ax3.grid(True)
ax3.set_ylabel('log X and Y');
plt.show()


# In[31]:


import matplotlib as mpl
mpl.rcParams['font.size'] = 11.
import matplotlib.pyplot as plt
import numpy as np
x = np.arange(11)
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(x, x);
ax2 = fig.add_subplot(312, sharex=ax1)
ax2.plot(2*x, 2*x);
ax3 = fig.add_subplot(313, sharex=ax1)
ax3.plot(3*x, 3*x);
plt.show()


# In[47]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
dates = [dt.datetime.today() + dt.timedelta(days=i)          for i in range(10)]
values = np.random.rand(len(dates))
plt.plot_date(mpl.dates.date2num(dates), values, linestyle='-');
plt.show()


# In[48]:


import matplotlib as mp
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
fig = plt.figure()
ax2 = fig.add_subplot(212)
date2_1 = dt.datetime(2008, 9, 23)
date2_2 = dt.datetime(2008, 10, 3)
delta2 = dt.timedelta(days=1)
dates2 = mpl.dates.drange(date2_1, date2_2, delta2)
y2 = np.random.rand(len(dates2))
ax2.plot_date(dates2, y2, linestyle='-');
dateFmt = mpl.dates.DateFormatter('%Y-%m-%d')
ax2.xaxis.set_major_formatter(dateFmt)
daysLoc = mpl.dates.DayLocator()
hoursLoc = mpl.dates.HourLocator(interval=6)
ax2.xaxis.set_major_locator(daysLoc)
ax2.xaxis.set_minor_locator(hoursLoc)
fig.autofmt_xdate(bottom=0.18)
fig.subplots_adjust(left=0.18)
ax1 = fig.add_subplot(211)
date1_1 = dt.datetime(2008, 9, 23)
date1_2 = dt.datetime(2009, 2, 16)
delta1 = dt.timedelta(days=10)
dates1 = mpl.dates.drange(date1_1, date1_2, delta1)
y1 = np.random.rand(len(dates1))
ax1.plot_date(dates1, y1, linestyle='-');
monthsLoc = mpl.dates.MonthLocator()
weeksLoc = mpl.dates.WeekdayLocator()
ax1.xaxis.set_major_locator(monthsLoc)
ax1.xaxis.set_minor_locator(weeksLoc)
monthsFmt = mpl.dates.DateFormatter('%b')
ax1.xaxis.set_major_formatter(monthsFmt)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[16]:





# In[ ]:





# In[ ]:





# In[ ]:




