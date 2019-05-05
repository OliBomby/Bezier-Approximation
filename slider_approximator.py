from shape_approximator import *
from slider_path import SliderPath
import time

pathTypeConversion = {'L': 'Linear',
                      'P': 'PerfectCurve',
                      'C': 'Catmull',
                      'B': 'Bezier'}

num_anchors = input("Number of anchors: ")
num_steps = input("Number of training steps: ")
num_testpoints = input("Evaluating resolution: ")
retarded = input("Retarded: ")

num_anchors = int(num_anchors) if num_anchors != "" else 500
num_steps = int(num_steps) if num_steps != "" else 10000
num_testpoints = int(num_testpoints) if num_testpoints != "" else 5000
retarded = float(retarded) if retarded != "" else 0

inp = input("Paste slider code here: ")
values = inp.split(',')
pathType = pathTypeConversion[values[5][0]]
path = (values[0] + ':' + values[1] + values[5][1:]).split('|')
path = np.vstack([vec(i) for i in path])

shape = SliderPath(pathType, path)
shape = np.vstack(shape.calculatedPath)

plt.ioff()
plt.ion()
plt.figure()
plt.show()

plot_alpha(shape)


firstTime = time.time()
loss, anchors = approximate_shape(shape, num_anchors, num_steps, num_testpoints, retarded)
print("Time took:", time.time() - firstTime)

# PrintSlider2(anchors, values, 1)
write_slider2(anchors, values, 1)
