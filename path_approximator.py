import numpy as np

BEZIER_TOLERANCE = 0.025
CATMULL_DETAIL = 50
CIRCULAR_ARC_TOLERANCE = 0.01


length_squared = lambda x: np.inner(x, x)


def approximate_bezier(control_points):
    output = []
    count = len(control_points)

    if count == 0:
        return output

    subdivision_buffer1 = np.empty([count, 2])
    subdivision_buffer2 = np.empty([count * 2 - 1, 2])

    to_flatten = []
    free_buffers = []

    to_flatten.append(control_points.copy())

    left_child = subdivision_buffer2

    while len(to_flatten) > 0:
        parent = to_flatten.pop()
        if bezier_is_flat_enough(parent):
            bezierApproximate(parent, output, subdivision_buffer1, subdivision_buffer2, count)

            free_buffers.append(parent)
            continue

        right_child = free_buffers.pop() if len(free_buffers) > 0 else np.empty([count, 2])
        bezier_subdivide(parent, left_child, right_child, subdivision_buffer1, count)

        for i in range(count):
            parent[i] = left_child[i]

        to_flatten.append(right_child)
        to_flatten.append(parent)

    output.append(control_points[-1].copy())
    return np.vstack(output)


def approximate_catmull(control_points):
    result = []

    for i in range(len(control_points) - 1):
        v1 = control_points[i - 1] if i > 0 else control_points[i]
        v2 = control_points[i]
        v3 = control_points[i + 1] if i < len(control_points) - 1 else v2 + v2 - v1
        v4 = control_points[i + 2] if i < len(control_points) - 2 else v3 + v3 - v2

        for c in range(CATMULL_DETAIL):
            result.append(catmull_find_point(v1, v2, v3, v4, c / CATMULL_DETAIL))
            result.append(catmull_find_point(v1, v2, v3, v4, (c + 1) / CATMULL_DETAIL))

    return result


def approximate_circular_arc(control_points):
    a = control_points[0]
    b = control_points[1]
    c = control_points[2]

    aSq = length_squared(b - c)
    bSq = length_squared(a - c)
    cSq = length_squared(a - b)

    if np.isclose(aSq, 0) or np.isclose(bSq, 0) or np.isclose(cSq, 0):
        return []

    s = aSq * (bSq + cSq - aSq)
    t = bSq * (aSq + cSq - bSq)
    u = cSq * (aSq + bSq - cSq)

    sum = s + t + u

    if np.isclose(sum, 0):
        return []

    centre = (s * a + t * b + u * c) / sum
    dA = a - centre
    dC = c - centre

    r = np.linalg.norm(dA)

    theta_start = np.arctan2(dA[1], dA[0])
    theta_end = np.arctan2(dC[1], dC[0])

    while theta_end < theta_start:
        theta_end += 2 * np.pi

    direction = 1
    theta_range = theta_range = theta_end - theta_start

    ortho_ato_c = c - a
    ortho_ato_c = np.array([ortho_ato_c[1], -ortho_ato_c[0]])
    if np.dot(ortho_ato_c, b - a) < 0:
        direction = -direction
        theta_range = 2 * np.pi - theta_range

    amount_points = 2 if 2 * r <= CIRCULAR_ARC_TOLERANCE else int(max(2, np.ceil(theta_range / (2 * np.arccos(1 - CIRCULAR_ARC_TOLERANCE / r)))))

    output = []

    for i in range(amount_points):
        fract = i / (amount_points - 1)
        theta = theta_start + direction * fract * theta_range
        o = np.array([np.cos(theta), np.sin(theta)]) * r
        output.append(centre + o)

    return output


def approximate_linear(control_points):
    result = []

    for c in control_points:
        result.append(c.copy())

    return result


def bezier_is_flat_enough(control_points):
    for i in range(1, len(control_points) - 1):
        p = control_points[i - 1] - 2 * control_points[i] + control_points[i + 1]
        if length_squared(p) > BEZIER_TOLERANCE * BEZIER_TOLERANCE * 4:
            return False
    return True


def bezier_subdivide(control_points, l, r, subdivision_buffer, count):
    midpoints = subdivision_buffer

    for i in range(count):
        midpoints[i] = control_points[i]

    for i in range(count):
        l[i] = midpoints[0].copy()
        r[count - i - 1] = midpoints[count - i - 1]

        for j in range(count - i - 1):
            midpoints[j] = (midpoints[j] + midpoints[j + 1]) / 2


def bezierApproximate(control_points, output, subdivision_buffer1, subdivision_buffer2, count):
    l = subdivision_buffer2
    r = subdivision_buffer1

    bezier_subdivide(control_points, l, r, subdivision_buffer1, count)

    for i in range(count - 1):
        l[count + i] = r[i + 1]

    output.append(control_points[0].copy())
    for i in range(1, count - 1):
        index = 2 * i
        p = 0.25 * (l[index - 1] + 2 * l[index] + l[index + 1])
        output.append(p.copy())


def catmull_find_point(vec1, vec2, vec3, vec4, t):
    t2 = t * t
    t3 = t * t2

    result = np.array([
        0.5 * (2 * vec2[0] + (-vec1[0] + vec3[0]) * t + (2 * vec1[0] - 5 * vec2[0] + 4 * vec3[0] - vec4[0]) * t2 + (-vec1[0] + 3 * vec2[0] - 3 * vec3[0] + vec4[0]) * t3),
        0.5 * (2 * vec2[1] + (-vec1[1] + vec3[1]) * t + (2 * vec1[1] - 5 * vec2[1] + 4 * vec3[1] - vec4[1]) * t2 + (-vec1[1] + 3 * vec2[1] - 3 * vec3[1] + vec4[1]) * t3)
    ])

    return result
