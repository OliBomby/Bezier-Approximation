import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.special.cython_special import binom
import time


times1 = []
times2 = []

matplotlib.use('TkAgg')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


def print_slider(anchors, plen=1, s=192, o=np.array([256, 192])):
    li = np.round(anchors * s + o, 0)
    for i in range(len(li) - 1):
        if (li[i] == li[i + 1]).all():
            li[i + 1] += 1
            i -= 2
    li = li.tolist()
    p1 = li.pop(0)
    ret = "B"
    for p in li:
        ret += "|" + str(int(p[0])) + ":" + str(int(p[1]))
    print("%s,%s,0,2,0,%s,1,%s" % (int(p1[0]), int(p1[1]), ret, plen * s))


def print_slider2(anchors, values, s):
    li = np.round(anchors * s, 0)
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
    f = open("slidercode.txt", "w+")
    f.write(",".join(values))
    f.close()


def write_slider(anchors, plen=1, s=192, o=np.array([256, 192])):
    li = np.round(anchors * s + o, 0)
    for i in range(len(li) - 1):
        if (li[i] == li[i + 1]).all():
            li[i + 1] += 1
            i -= 2
    li = li.tolist()
    p1 = li.pop(0)
    ret = "B"
    for p in li:
        ret += "|" + str(int(p[0])) + ":" + str(int(p[1]))

    with open("slidercode.txt", "w+") as f:
        f.write("%s,%s,0,2,0,%s,1,%s" % (int(p1[0]), int(p1[1]), ret, plen * s))

    print("Successfully saved slidercode to slidercode.txt")


def write_slider2(anchors, values, s):
    li = np.round(anchors * s, 0)
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

    with open("slidercode.txt", "w+") as f:
        f.write(",".join(values))

    print("Successfully saved slidercode to slidercode.txt")


def print_anchors(anchors):
    ret = ""
    for p in anchors:
        ret += "|" + str(p[0]) + ":" + str(p[1])
    print(ret)


def plot(ll, a, p, l):
    ax1.cla()
    ax1.plot(ll)

    ax2.cla()
    ax2.axis('equal')
    ax2.plot(p[:, 0], p[:, 1], color="green")
    # ax2.plot(a[:, 0], a[:, 1], color="red")

    a = distance_array(l)
    ax3.cla()
    ax3.plot(a, color="red")

    a = distance_array(p)
    ax4.cla()
    ax4.plot(a, color="green")

    plt.draw()
    plt.pause(0.0001)


def plot_alpha(l):
    a = np.clip(30 / len(l), 0, 1)
    ax2.cla()
    ax2.axis('equal')
    ax2.scatter(l[:, 0], l[:, 1], color='green', alpha=a, marker='.')

    plt.draw()
    plt.pause(0.0001)


def plot_vel_distr(l):
    a = distance_array(l)
    ax3.cla()
    ax3.plot(a)

    plt.draw()
    plt.pause(0.0001)


def vec(b):
    return np.array(b.split(':'), dtype=np.float32)


def total_length(shape):
    return np.sum(norm(np.diff(shape, axis=0), axis=1))


def distance_array(shape):
    return norm(np.diff(shape, axis=0), axis=1)


def bezier(anchors, num_points):
    w = generate_weights(anchors.shape[0], num_points)
    return np.matmul(w, anchors)


def powers(n, m):
    pwrs = [1]
    for b in range(n):
        pwrs.append(pwrs[b] * m)
    return pwrs


def pathify2(pred, template):
    num = pred.shape[0]
    pred = np.vstack([template[0], pred, template[-1]])
    pred_d = distance_array(pred)
    pred_sum = np.sum(pred_d)

    num_tmpl = len(template)
    points = np.empty([num, 2])
    sprog = 0
    for i in range(num):
        sprog += pred_d[i]

        div = sprog / pred_sum * (num_tmpl - 1)
        f = int(np.floor(div))
        c = int(np.ceil(div))
        r = div - f

        pf = template[f]
        pc = template[c]
        diff = pc - pf

        points[i] = pf + diff * r
    return points


def pathify2_with_endpoints(pred, template):
    num = pred.shape[0]
    pred_d = np.concatenate((np.zeros(1), distance_array(pred)))
    # pred_sum = np.sum(pred_d)
    pred_sum = 0
    for d in pred_d:
        pred_sum += d

    num_tmpl = len(template)
    points = np.empty([num, 2])
    sprog = 0
    for i in range(num):
        sprog += pred_d[i]

        div = sprog / pred_sum * (num_tmpl - 1)
        f = int(np.floor(div))
        c = int(np.ceil(div))
        r = div - f

        pf = template[f]
        pc = template[c]
        diff = pc - pf

        points[i] = pf + diff * r
    return points


def pathify(pred, shape, shape_d, shape_l):
    num = pred.shape[0]
    pred = np.vstack([shape[0], pred, shape[-1]])
    pred_d = distance_array(pred)
    return shape_from_distances(pred_d, shape, shape_d, shape_l, num)


def shape_from_distances(dists, shape, shape_d, shape_l, num=None):
    if num is None:
        num = len(dists)
    dist = np.sum(dists)
    points = np.empty([num, 2])
    sprog = 0
    s_index = 0
    slen = shape_d[s_index]
    for i in range(num):
        sprog += dists[i] / dist * shape_l

        while sprog > slen + 1E-6:
            sprog -= slen
            s_index += 1
            if s_index > len(shape_d) - 1:
                s_index = len(shape) - 2
                sprog = slen
                break
            slen = shape_d[s_index]

        points[i] = shape[s_index] + (shape[s_index + 1] - shape[s_index]) * sprog / slen

    return points


def generate_weights(num_anchors, num_testpoints):
    if num_anchors > 1000:
        return generate_weights2(num_anchors, num_testpoints)

    order = num_anchors - 1
    w = np.zeros([num_testpoints, num_anchors])
    binoms = np.array([binom(order, k) for k in range(num_anchors)])
    for i in range(num_testpoints):
        t = (i + 0.5) / num_testpoints
        m1 = np.array(powers(order, 1 - t)[::-1])
        m2 = np.array(powers(order, t))
        w[i, :] = binoms * m1 * m2
    return w


# This function is more numerically stable for high anchor count
def generate_weights2(num_anchors, num_testpoints):
    n = num_anchors - 1
    w = np.zeros([num_testpoints, num_anchors])
    for i in range(num_testpoints):
        t = (i + 0.5) / num_testpoints

        middle = int(round(t * n))
        cm = get_weight(middle, n, t)
        w[i, middle] = cm

        c = cm
        for k in range(middle, n):
            c = c * (n - k) / (k + 1) / (1 - t) * t  # Move right
            w[i, k + 1] = c
            if c == 0:
                break

        c = cm
        for k in range(middle - 1, -1, -1):
            c = c / (n - k) * (k + 1) * (1 - t) / t  # Move left
            w[i, k] = c
            if c == 0:
                break

    return w


def generate_weights_with_endpoints(num_anchors, num_testpoints):
    order = num_anchors - 1
    w = np.zeros([num_testpoints, num_anchors])
    binoms = np.array([binom(order, k) for k in range(num_anchors)])
    for i in range(num_testpoints):
        t = i / (num_testpoints - 1)
        m1 = np.array(powers(order, 1 - t)[::-1])
        m2 = np.array(powers(order, t))
        w[i, :] = binoms * m1 * m2
    return w


def get_weight(anchor, n, t):
    ntm = 0
    ntp = 0
    b = anchor
    if b > n // 2:
        b = n - b
    cm = 1
    for i in range(b):
        cm *= n - i
        cm /= i + 1
        while cm > 1 and ntm < (n - anchor):
            cm *= (1 - t)
            ntm += 1
        while cm > 1 and ntp < anchor:
            cm *= t
            ntp += 1

    cm = cm * (1 - t) ** (n - anchor - ntm) * t ** (anchor - ntp)
    return cm


def test_anchors(anchors, shape, num_testpoints):
    shape_d = distance_array(shape)
    shape_l = np.sum(shape_d)

    new_shape = bezier(anchors, num_testpoints)
    labels = pathify(new_shape, shape, shape_d, shape_l)
    distances = norm(labels - new_shape, axis=1)
    loss = np.mean(np.square(distances))
    print("loss: %s" % loss)


def plot_distribution(anchors, shape):
    shape_d = distance_array(shape)
    shape_l = np.sum(shape_d)

    reduced_shape = bezier(anchors, 1000)
    reduced_labels = pathify(reduced_shape, shape, shape_d, shape_l)
    plot_alpha(reduced_labels)


def approximate_shape(shape, num_anchors, num_steps=5000, num_testpoints=1000, retarded=0):
    # Generate the weights for the bezier conversion
    print("Generating weights...")
    w = generate_weights(num_anchors, num_testpoints)
    shape_d = distance_array(shape)
    shape_l = np.sum(shape_d)

    # Generate pathify template
    # Means the same target shape but with equal spacing, so pathify runs in linear time
    print("Initializing pathify template...")
    num_templatepoints = 1000
    template_distance = shape_l / (num_templatepoints - 1)
    dists = [0]
    for i in range(num_templatepoints - 1):
        dists.append(template_distance)
    template = shape_from_distances(dists, shape, shape_d, shape_l)

    # Initialize the anchors
    print("Initializing anchors...")
    dists = []
    last = 0
    for i in range(num_anchors):
        t = i / (num_anchors - 1)
        dists.append(t - last)
        last = t

    anchors = shape_from_distances(dists, shape, shape_d, shape_l)
    firstanchor = anchors[0]
    lastanchor = anchors[-1]
    middleanchors = anchors[1:-1]

    num_trainable_anchors = num_anchors - 2

    # Scamble this shit
    middleanchors += np.random.rand(num_trainable_anchors, 2) * retarded

    # Initialize the labels
    print("Initializing test points...")
    dists = []
    last = 0
    for i in range(num_testpoints):
        t = (i + 0.5) / num_testpoints
        dists.append(t - last)
        last = t

    labels = shape_from_distances(dists, shape, shape_d, shape_l)

    # Calculate initial loss
    print("Calculating initial loss...")
    predictions = np.matmul(w, anchors)
    distances = norm(labels - predictions, axis=1)
    lossq = np.sqrt(np.mean(np.square(distances)))

    # This is the computational graph
    print("Building the computational graph...")
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(lossq, global_step,
                                               1000, 0.9, staircase=False)

    # Labels
    labelstensor = tf.placeholder(tf.float32, shape=[num_testpoints, 2])

    # Anchors
    firstanchortensor = tf.constant(firstanchor, shape=[1, 2], dtype=tf.float32)
    lastanchortensor = tf.constant(lastanchor, shape=[1, 2], dtype=tf.float32)
    middleanchorstensor = tf.Variable(middleanchors, [num_trainable_anchors, 2], dtype=tf.float32)
    anchors = tf.concat([firstanchortensor, middleanchorstensor, lastanchortensor], axis=0)

    # Computation
    weights = tf.constant(w, dtype=tf.float32)
    predictions = tf.matmul(weights, anchors)

    # Loss
    loss = tf.losses.mean_squared_error(labelstensor, predictions)

    # Train step
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Training loop
    last_anchors = None
    last_labels = labels
    _loss = 0
    with tf.Session() as sess:
        print("Initializing global variables...")
        sess.run(tf.global_variables_initializer())
        loss_list = []

        print("Starting training loop")
        for step in range(num_steps):
            _loss, _predictions, _anchors, _train_step, _learning_rate = sess.run(
                [loss, predictions, anchors, train_step, learning_rate],
                feed_dict={labelstensor: last_labels})

            if np.isnan(_loss):
                break

            if step % 100 == 0:
                # last_labels = pathify(_predictions, shape, shape_d, shape_l)
                last_labels = pathify2(_predictions, template)

            if step % 1000 == 0:
                loss_list.append(np.log(_loss))
                print("Step ", step, "Loss ", _loss, "Leaning rate ", _learning_rate)
                plot(loss_list, _anchors, _predictions, last_labels)

            if _loss < 1e-11:
                break

            last_anchors = _anchors

    print("Final loss: ", _loss)
    return _loss, last_anchors


def approximate_shape2(shape, num_anchors, num_steps=5000, num_testpoints=1000):
    # Generate the weights for the bezier conversion
    print("Generating weights...")
    w = generate_weights_with_endpoints(num_anchors, num_testpoints)
    shape_d = distance_array(shape)
    shape_l = np.sum(shape_d)

    # Generate pathify template
    # Means the same target shape but with equal spacing, so pathify runs in linear time
    print("Initializing pathify template...")
    num_templatepoints = 1000
    template_distance = shape_l / (num_templatepoints - 1)
    dists = [0]
    for i in range(num_templatepoints - 1):
        dists.append(template_distance)
    template = shape_from_distances(dists, shape, shape_d, shape_l)

    # Initialize the anchors
    print("Initializing test points with reasonable velocity distribution...")
    dists = []
    last = 0
    for i in range(num_anchors):
        t = i / (num_anchors - 1)
        dists.append(t - last)
        last = t

    anchors = shape_from_distances(dists, shape, shape_d, shape_l)
    points = np.matmul(w, anchors)

    labels = pathify2_with_endpoints(points, template)

    # Calculate least squares matrix
    print("Calculating least squares matrix...")
    wt = np.transpose(w)
    # least_squares_matrix = np.matmul(np.linalg.inv(np.matmul(wt, w)), wt)
    arr = np.matmul(wt, w)

    # Training loop
    anchors = None
    loss = 0
    loss_list = []

    print("Starting training loop")
    for step in range(num_steps):
        # anchors = np.matmul(least_squares_matrix, labels)
        anchors = np.linalg.solve(arr, np.matmul(wt, labels))
        points = np.matmul(w, anchors)
        diff = labels - points
        loss = np.mean(np.square(diff))

        loss_list.append(np.log(loss))
        print("Step ", step, "Loss ", loss, "Leaning rate ", 1)
        plot(loss_list, anchors, points, labels)

        labels = pathify2_with_endpoints(points, template)

    print("Final loss: ", loss)
    return loss, anchors


if __name__ == "__main__":
    plt.ioff()
    plt.ion()
    plt.figure()
    plt.show()

    num_anchors = 500
    num_steps = 20000
    num_testpoints = 5000
    num_templatepoints = 1000

    from shapes import GosperCurve
    shape = GosperCurve(1)
    shape = shape.make_shape(1)

    firstTime = time.time()
    loss, anchors = approximate_shape(shape, num_anchors, num_steps, num_testpoints, retarded=50)
    print("Time took:", time.time() - firstTime)
    print("Times 1:", np.mean(times1))
    print("Times 2:", np.mean(times2))

    ##PrintSlider(anchors, length(shape))
    write_slider(anchors, total_length(shape))

    test_anchors(anchors, shape, 10000)
    plot_distribution(anchors, shape)
