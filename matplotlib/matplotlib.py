#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  5 06:38:10 2025

@author: rajesh
"""

'''
Matplotlib is an open-source visualization library for the Python programming language, widely used for creating static, animated and interactive plots. 
With Matplotlib, we can perform a wide range of visualization tasks, including:

    Creating basic plots such as line, bar and scatter plots.
    Customizing plots with labels, titles, legends and color schemes.
    Adjusting figure size, layout and aspect ratios.
    Saving plots in various formats like PNG, PDF and SVG.
    Combining multiple plots into subplots for better data representation.
    Creating interactive plots using the widget module.



'''

import matplotlib.pyplot as plt

#basic line chart
x = [0, 1, 2, 3, 4]
y = [0, 1, 4, 9, 16]

plt.plot(x, y)
plt.show()

'''
The parts of a Matplotlib figure include :

    Figure: The overarching container that holds all plot elements, acting as the canvas for visualizations.
    Axes: The areas within the figure where data is plotted; each figure can contain multiple axes.
    Axis: Represents the x-axis and y-axis, defining limits, tick locations, and labels for data interpretation.
    Lines and Markers: Lines connect data points to show trends, while markers denote individual data points in plots like scatter plots.
    Title and Labels: The title provides context for the plot, while axis labels describe what data is being represented on each axis.

'''


import matplotlib.pyplot as plt
import numpy as np

x = [0, 2, 4, 6, 8]
y = [0, 4, 16, 36, 64]

fig, ax = plt.subplots()  
ax.plot(x, y, marker='o', label="Data Points")

ax.set_title("Basic Components of Matplotlib Figure")
ax.set_xlabel("X-Axis") 
ax.set_ylabel("Y-Axis")  


plt.show()





'''
Matplotlib offers a wide range of plot types to suit various data visualization needs. Here are some of the most commonly used types of plots in Matplotlib:

    1. Line Graph
    2. Bar Chart
    3. Histogram
    4. Scatter Plot
    5. Pie Chart
    6. 3D Plot

'''




# Define X and Y variable data
x = np.array([1, 2, 3, 4])
y = x*2

plt.plot(x, y)
plt.xlabel("X-axis")  # add X-axis label
plt.ylabel("Y-axis")  # add Y-axis label
plt.title("Any suitable title")  # add title
plt.show()


#multiple plots on the data

x = np.array([1, 2, 3, 4])
y = x*2

# first plot with X and Y data
plt.plot(x, y)

x1 = [2, 4, 6, 8]
y1 = [3, 5, 7, 9]

# second plot with x1 and y1 data
plt.plot(x1, y1, '-.')

plt.xlabel("X-axis data")
plt.ylabel("Y-axis data")
plt.title('multiple plots')
plt.show()


#zip
x = [1, 2, 3]
y = ['a', 'b', 'c']
list(zip(x, y))  # → [(1, 'a'), (2, 'b'), (3, 'c')]

#enumerate
x = [1, 2, 3]
y = ['a', 'b', 'c']

for i, (a, b) in enumerate(zip(x, y)):
    print(i, a, b)

#Annotations
# Sample data
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Create a line chart
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o')

# Add annotations
for i, (xi, yi) in enumerate(zip(x, y)):
    plt.annotate(f'({xi}, {yi})', (xi, yi), textcoords="offset points", xytext=(0, 20), ha='center')

# Add title and labels
plt.title('Line Chart with Annotations')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')

# Display grid
plt.grid(True)

# Show the plot
plt.show()


#bar chart

fruits = ['Apples', 'Bananas', 'Cherries', 'Dates']
sales = [400, 350, 300, 450]

plt.bar(fruits, sales, color='red')
plt.title('Fruit Sales')
plt.xlabel('Fruits')
plt.ylabel('Sales')
plt.show()

#horizontal bar charts
cars = ['Audi', 'BMW', 'Skoda', 'Volkswagen']
sales = [400, 350, 300, 450]

plt.barh(cars, sales)

plt.title('Car Sales')
plt.xlabel('Cars')
plt.ylabel('Sales')
plt.show()



fruits = ['Apples', 'Bananas', 'Cherries', 'Dates']
sales = [400, 350, 300, 450]

plt.bar(fruits, sales, width=0.3)
plt.title('Fruit Sales')
plt.xlabel('Fruits')
plt.ylabel('Sales')
plt.show()



#Scatter plots


x = np.array([12, 45, 7, 32, 89, 54, 23, 67, 14, 91])
y = np.array([99, 31, 72, 56, 19, 88, 43, 61, 35, 77])

plt.scatter(x, y)
plt.show()


#Multi scatte rplot
x1 = np.array([160, 165, 170, 175, 180, 185, 190, 195, 200, 205])
y1 = np.array([55, 58, 60, 62, 64, 66, 68, 70, 72, 74])

x2 = np.array([150, 155, 160, 165, 170, 175, 180, 195, 200, 205])
y2 = np.array([50, 52, 54, 56, 58, 64, 66, 68, 70, 72])

plt.scatter(x1, y1, color='blue', label='Group 1')
plt.scatter(x2, y2, color='red', label='Group 2')

plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Comparison of Height vs Weight between two groups')

plt.legend()
plt.show()


#bubble plot
# Data
x_values = [1, 2, 3, 4, 5]
y_values = [2, 3, 5, 7, 11]
bubble_sizes = [30, 80, 150, 200, 300]

# Create a bubble chart with customization
plt.scatter(x_values, y_values, s=bubble_sizes, alpha=0.6, edgecolors='b', linewidths=2)

# Add title and axis labels
plt.title("Bubble Chart with Transparency")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Display the plot
plt.show()

#Pie plots
cars = ['AUDI', 'BMW', 'FORD',
        'TESLA', 'JAGUAR', 'MERCEDES']

data = [23, 17, 35, 29, 12, 41]

# Creating plot
fig = plt.figure(figsize=(10, 7))
plt.pie(data, labels=cars)

# show plot
plt.show()



#creating a 3d plot

from mpl_toolkits import mplot3d

fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining all 3 axis
z = np.linspace(0, 1, 100)
x = z * np.sin(25 * z)
y = z * np.cos(25 * z)
 
# plotting
ax.plot3D(x, y, z, 'green')
ax.set_title('3D line plot geeks for geeks')
plt.show()

#Heatmaps
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt 
    
# generating 2-D 10x10 matrix of random numbers 
# from 1 to 100 
data = np.random.randint(low=1, 
                         high=100, 
                         size=(10, 10)) 
    
# plotting the heatmap 
hm = sns.heatmap(data=data, 
                annot=True) 
    
# displaying the plotted heatmap 
plt.show()


