import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.interpolate import interp1d
from scipy.special import comb
import time


matplotlib.use('TkAgg')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)


def encode_anchors(anchors, s=1, o=np.zeros(0)):
    li = np.round(anchors * s + o)
    for i in range(len(li) - 1):
        if (li[i] == li[i + 1]).all():
            li[i + 1] += 1
            i -= 2
    li = li.tolist()
    p1 = li.pop(0)
    ret = "B"
    for p in li:
        ret += "|" + str(int(p[0])) + ":" + str(int(p[1]))
    return p1, ret


def write_slider(anchors, plen=1, s=192, o=np.array([256, 192])):
    p1, ret = encode_anchors(anchors, s, o)

    with open("slidercode.txt", "w+") as f:
        f.write("%s,%s,0,2,0,%s,1,%s" % (int(p1[0]), int(p1[1]), ret, plen * s))

    print("Successfully saved slidercode to slidercode.txt")


def write_slider2(anchors, values, s=1):
    p1, ret = encode_anchors(anchors, s)
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


def plot_points(p):
    ax2.cla()
    ax2.axis('equal')
    ax2.plot(p[:, 0], p[:, 1], color="green")


def plot(ll, a, p, l):
    ax1.cla()
    ax1.plot(ll)
    ax1.set_yscale('log')

    ax2.cla()
    ax2.axis('equal')
    ax2.plot(p[:, 0], p[:, 1], color="green")
    if a is not None:
        ax2.plot(a[:, 0], a[:, 1], color="red")

    a = distance_array(l)
    ax3.cla()
    ax3.plot(a, color="red")

    a = distance_array(p)
    ax4.cla()
    ax4.plot(a, color="green")

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


def dist_cumsum(shape):
    return np.pad(np.cumsum(distance_array(shape)), (1, 0))


def bezier(anchors, num_points):
    w = generate_weights(anchors.shape[0], num_points)
    return np.matmul(w, anchors)


def pathify3_with_endpoints(pred, template):
    pred_d = np.concatenate((np.zeros(1), distance_array(pred)))
    pred_cumsum = np.cumsum(pred_d)
    progs = pred_cumsum / pred_cumsum[-1]
    points = template(progs)
    return points


def generate_weights_from_t(t):
    binoms = comb(num_anchors - 1, np.arange(num_anchors))
    p = np.power(t[:, np.newaxis], np.arange(num_anchors))
    w = binoms * p[::-1, ::-1] * p
    return w


def generate_weights(num_anchors, num_testpoints):
    if num_anchors > 1000:
        return generate_weights2(num_anchors, num_testpoints)

    t = np.linspace(0.5 / num_testpoints, 1 - 0.5 / num_testpoints, num_testpoints)
    return generate_weights_from_t(t)


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
    if num_anchors > 1000:
        return generate_weights2_with_endpoints(num_anchors, num_testpoints)

    t = np.linspace(0, 1, num_testpoints)
    return generate_weights_from_t(t)


def generate_weights2_with_endpoints(num_anchors, num_testpoints):
    n = num_anchors - 1
    w = np.zeros([num_testpoints, num_anchors])
    for i in range(num_testpoints):
        t = i / (num_testpoints - 1)

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

def anchor_positions_on_curve(anchors):
    num_anchors = len(anchors)
    n = num_anchors - 1
    positions = []
    for i in range(num_anchors):
        t = i / n
        cm = get_weight(i, n, t)
        pos = anchors[i] * cm

        c = cm
        for k in range(i, n):
            c = c * (n - k) / (k + 1) / (1 - t) * t  # Move right
            pos += anchors[k + 1] * c
            if c == 0:
                break

        c = cm
        for k in range(i - 1, -1, -1):
            c = c / (n - k) * (k + 1) * (1 - t) / t  # Move left
            pos += anchors[k] * c
            if c == 0:
                break

        positions.append(pos)

    return positions


def get_interpolator(shape):
    shape_d_cumsum = dist_cumsum(shape)
    return interp1d(shape_d_cumsum / shape_d_cumsum[-1], shape, axis=0, assume_sorted=True, copy=False)


def test_anchors(anchors, shape, num_testpoints):
    new_shape = bezier(anchors, num_testpoints)
    labels = pathify3_with_endpoints(new_shape, get_interpolator(shape))
    loss = np.mean(np.square(labels - new_shape))
    print("loss: %s" % loss)


def plot_distribution(anchors, shape):
    reduced_shape = bezier(anchors, 1000)
    reduced_labels = pathify3_with_endpoints(reduced_shape, get_interpolator(shape))
    plot_alpha(reduced_labels)


def approximate_shape3(shape, num_anchors, num_steps=5000, num_testpoints=1000, retarded=0):
    # Generate the weights for the bezier conversion
    print("Generating weights...")
    w = generate_weights_with_endpoints(num_anchors, num_testpoints)
    wt = np.transpose(w)

    # Generate pathify template
    # Means the same target shape but with equal spacing, so pathify runs in linear time
    print("Initializing pathify template...")
    template = get_interpolator(shape)

    # Initialize the anchors
    print("Initializing test points with reasonable velocity distribution...")
    anchors = template(np.linspace(0, 1, num_anchors))
    points = np.matmul(w, anchors)
    labels = pathify3_with_endpoints(points, template)

    # Scamble this shit
    if retarded > 0:
        random_offset = np.random.rand(num_anchors, 2) * retarded
        random_offset[0, :] = 0
        random_offset[-1, :] = 0
        anchors += random_offset

    # Set up adam optimizer parameters
    learning_rate = 8
    b1 = 0.9
    b2 = 0.92
    epsilon = 1E-8
    m = np.zeros(anchors.shape)
    v = np.zeros(anchors.shape)

    # Set up mask for constraining endpoints
    learnable_mask = np.zeros(anchors.shape)
    learnable_mask[1:-1] = 1

    # Training loop
    loss_list = []
    step = 0
    print("Starting training loop")
    for step in range(1, num_steps):
        points = np.matmul(w, anchors)

        if step % 11 == 0:
            labels = pathify3_with_endpoints(points, template)

        diff = labels - points
        loss = np.mean(np.square(diff))

        # Calculate gradients
        grad = -1 / num_anchors * np.matmul(wt, diff)

        # Apply learnable mask
        grad *= learnable_mask

        # Update with adam optimizer
        m = b1 * m + (1 - b1) * grad
        v = b2 * v + (1 - b2) * np.square(grad)
        m_corr = m / (1 - b1**step)
        v_corr = v / (1 - b2**step)
        anchors -= learning_rate * m_corr / (np.sqrt(v_corr) + epsilon)

        # Logging
        loss_list.append(loss)

        # if step % 100 == 0:
        #     print("Step ", step, "Loss ", loss, "Rate ", learning_rate)
        #     plot(loss_list, anchors, points, labels)

        if loss < 0.01:
            break

    points = np.matmul(w, anchors)
    loss = np.mean(np.square(labels - points))

    # plot(loss_list, anchors, points, labels)
    print("Final loss: ", loss, step + 1)
    return loss, anchors

if __name__ == "__main__":
    # plt.ion()
    # plt.show()

    num_anchors = 10
    num_steps = 10000
    num_testpoints = 200

    from shapes import CircleArc
    shape = CircleArc(np.zeros(2), 100, 0, 2 * np.pi)
    shape = shape.make_shape(100)
    # from shapes import GosperCurve
    # shape = GosperCurve(1)
    # shape = shape.make_shape(1)

    firstTime = time.time()
    loss, anchors = approximate_shape3(shape, num_anchors, num_steps, num_testpoints)
    print("Time took:", time.time() - firstTime)

    ##PrintSlider(anchors, length(shape))
    write_slider(anchors, total_length(shape))

    test_anchors(anchors, shape, 10000)
    # plot_distribution(anchors, shape)
    # plt.pause(1000)
