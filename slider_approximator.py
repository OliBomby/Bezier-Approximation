from shape_approximator import *
from slider_path import SliderPath
import time
from structs import Poi, array_to_poi, get_smallest_angle

pathTypeConversion = {'L': 'Linear',
                      'P': 'PerfectCurve',
                      'C': 'Catmull',
                      'B': 'Bezier'}


def convert_slider(shape, num_anchors, num_steps, num_testpoints, retarded):
    plot_alpha(shape)

    firstTime = time.time()
    loss, anchors = approximate_shape(shape, num_anchors, num_steps, num_testpoints, retarded)
    # loss, anchors = approximate_shape2(shape, num_anchors, num_steps, num_testpoints)
    # loss, anchors = approximate_shape3(shape, num_anchors, num_steps, num_testpoints)
    print("Time took:", time.time() - firstTime)
    return loss, anchors


def main():
    num_anchors = input("Number of anchors: ")
    num_steps = input("Number of training steps: ")
    num_testpoints = input("Evaluating resolution: ")
    retarded = input("Retarded: ")

    num_anchors = int(num_anchors) if num_anchors != "" else 500
    num_steps = int(num_steps) if num_steps != "" else 10000
    num_testpoints = int(num_testpoints) if num_testpoints != "" else 5000
    retarded = float(retarded) if retarded != "" else 0

    # inp = input("Paste slider code here: ")

    lines = []
    with open("input.txt", "r") as f:
        lines = f.readlines()

    inp = lines[0][3:]
    print(inp)

    values = inp.split(',')
    pathType = pathTypeConversion[values[5][0]]
    path = (values[0] + ':' + values[1] + values[5][1:]).split('|')
    path = np.vstack([vec(i) for i in path])

    shape = SliderPath(pathType, path)
    shape = np.vstack(shape.calculatedPath)

    loss, anchors = convert_slider(shape, num_anchors, num_steps, num_testpoints, retarded)

    # PrintSlider2(anchors, values, 1)
    write_slider2(anchors, values, 1)


def main2():
    lines = []
    with open("input2.txt", "r") as f:
        lines = f.readlines()

    hitobjects = []
    at = False
    for l in lines:
        ls = l.strip()
        if not at:
            if ls == "[HitObjects]":
                at = True
            continue

        if ls == "":
            continue

        hitobjects.append(ls)

    hitobjects = hitobjects[len(hitobjects) - 15:]
    with open("slidercode.txt", "w+") as f:
        f.write("[HitObjects]\n")

    for ho in hitobjects:
        values = ho.split(',')

        if values[3] != "2" and values[3] != "6" or len(values) < 8:
            with open("slidercode.txt", "a") as f:
                f.write(ho + "\n")
            continue

        pathType = pathTypeConversion[values[5][0]]
        path = (values[0] + ':' + values[1] + values[5][1:]).split('|')
        path = np.vstack([vec(i) for i in path])

        shape = SliderPath(pathType, path)
        shape = np.vstack(shape.calculatedPath)

        reds = 0
        anchors = len(path)
        for i in range(1, anchors - 2):
            if (path[i] == path[i + 1]).all():
                if abs(get_smallest_angle(array_to_poi(path[i] - path[i - 1]).getAngle(),
                                          array_to_poi(path[i + 2] - path[i + 1]).getAngle())) > 0.1:
                    reds += 1
                else:
                    reds += 0.2

        prev_a = None
        total_angle = 0
        for i in range(len(shape) - 1):
            diff = shape[i + 1] - shape[i]
            a = array_to_poi(diff).getAngle()
            if prev_a is None:
                prev_a = a
            else:
                total_angle += abs(prev_a - a)
            prev_a = a

        num_anchors = int(2 + np.ceil(total_angle * 1.13) + reds * 50)
        num_anchors = min(num_anchors, 10000)
        print("num_anchors: %s" % num_anchors)

        loss, anchors = convert_slider(shape, num_anchors, 10000, 5000, 0)

        li = np.round(anchors * 1, 0)
        for i in range(len(li) - 1):
            if (li[i] == li[i + 1]).all():
                li[i + 1] += 1
                i -= 2
        li = li.tolist()
        p1 = li.pop(0)
        ret = "B"
        for p in li:
            ret += "|" + str(int(p[0])) + ":" + str(int(p[1]))
        values[0] = str(int(p1[0]))
        values[1] = str(int(p1[1]))
        values[5] = ret

        with open("slidercode.txt", "a") as f:
            f.write(",".join(values) + "\n")


main()
# plt.waitforbuttonpress()
plt.pause(999999)
