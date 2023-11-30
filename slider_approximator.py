from shape_approximator import *
from slider_path import SliderPath
from structs import array_to_poi, get_smallest_angle


pathTypeConversion = {'L': 'Linear',
                      'P': 'PerfectCurve',
                      'C': 'Catmull',
                      'B': 'Bezier'}


def convert_slider(shape, num_anchors, args):
    if args.plot:
        plot_alpha(shape)

    firstTime = time.time()
    if args.mode == "bspline":
        loss, anchors = piecewise_linear_to_bspline(shape, args.order, num_anchors, args.num_steps, args.num_testpoints,
                                                   args.retarded, args.learning_rate, args.b1, args.b2, not args.silent, args.plot)
    else:
        loss, anchors = piecewise_linear_to_bezier(shape, num_anchors, args.num_steps, args.num_testpoints,
                                                   args.retarded, args.learning_rate, args.b1, args.b2, not args.silent, args.plot)

    if not args.silent:
        print("Time took:", time.time() - firstTime)

    return loss, anchors


def get_shape(values):
    pathType = pathTypeConversion[values[5][0]]
    control_points = (values[0] + ':' + values[1] + values[5][1:]).split('|')
    control_points = np.vstack([vec(i) for i in control_points])

    shape = SliderPath(pathType, control_points)
    shape = np.vstack(shape.calculatedPath)
    return shape, control_points


def determine_control_point_count(shape, control_points, args):
    reds = 0
    anchors = len(control_points)
    for i in range(1, anchors - 2):
        if (control_points[i] == control_points[i + 1]).all():
            if abs(get_smallest_angle(array_to_poi(control_points[i] - control_points[i - 1]).getAngle(),
                                      array_to_poi(control_points[i + 2] - control_points[i + 1]).getAngle())) > 0.1:
                reds += 1
            else:
                reds += 0.2

    prev_a = None
    total_angle = 0
    for i in range(len(shape) - 1):
        diff = shape[i + 1] - shape[i]
        a = array_to_poi(diff).getAngle()
        if prev_a is not None:
            total_angle += abs(prev_a - a)
        prev_a = a

    num_anchors = int(2 + np.ceil(total_angle * 1.13) + reds * (min(50, args.order) if args.mode == "bspline" else 50))
    num_anchors = min(num_anchors, 10000)
    return num_anchors


def main(args):
    if args.slidercode is not None:
        inp = args.slidercode
    else:
        with open(args.input, "r") as f:
            lines = f.readlines()
        inp = lines[0][3:]

    if not args.silent:
        print(inp)
    values = inp.split(',')
    shape, control_points = get_shape(values)

    num_anchors = args.num_anchors if args.num_anchors is not None else determine_control_point_count(shape, control_points, args)
    if not args.silent:
        print("num_anchors: %s" % num_anchors)

    loss, anchors = convert_slider(shape, num_anchors, args)

    if args.print_output:
        print_slider2(anchors, values, 1)
    else:
        write_slider2(anchors, values, 1, args.output, not args.silent)


def main2(args):
    with open(args.input, "r") as f:
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
    with open(args.output, "w+") as f:
        f.write("[HitObjects]\n")

    for ho in hitobjects:
        values = ho.split(',')

        if values[3] != "2" and values[3] != "6" or len(values) < 8:
            with open(args.output, "a") as f:
                f.write(ho + "\n")
            continue

        shape, control_points = get_shape(values)

        num_anchors = determine_control_point_count(shape, control_points, args)
        if not args.silent:
            print("num_anchors: %s" % num_anchors)

        loss, anchors = convert_slider(shape, num_anchors, args)

        p1, ret = encode_anchors(anchors)
        values[0] = str(int(p1[0]))
        values[1] = str(int(p1[1]))
        values[5] = ret

        with open(args.output, "a") as f:
            f.write(",".join(values) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--slidercode", type=str, default=None)
    parser.add_argument("--input", type=str, default="input.txt")
    parser.add_argument("--output", type=str, default="slidercode.txt")
    parser.add_argument("--num-anchors", type=int, default=None)
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--num-testpoints", type=int, default=1000)
    parser.add_argument("--retarded", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=4)
    parser.add_argument("--b1", type=float, default=0.8)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--full-map", type=bool, default=False)
    parser.add_argument("--silent", type=bool, default=False)
    parser.add_argument("--print-output", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="bezier", choices=["bezier", "bspline"])
    parser.add_argument("--order", type=int, default=3)
    parser.add_argument("--plot", type=bool, default=False)
    args = parser.parse_args()

    if args.plot:
        init_plot()

    if args.full_map:
        main2(args)
    else:
        main(args)
