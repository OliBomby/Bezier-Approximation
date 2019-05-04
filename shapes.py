from structs import *
import numpy as np
import gosper


def generate_shape(func, mini, maxi, step):
    shape = []
    t = mini
    while t <= maxi + 0.00001:  # Extra to account for rounding error
        point = func(t)
        shape.append(point)
        t += step

    return np.vstack(shape)


class ArchimedeanSpiral:
    def __init__(self, a, b, rotations):
        self.a = a
        self.b = b
        self.rotations = rotations

    def make_shape(self, num):
        delta_theta = 2 * np.pi / num
        theta_max = self.rotations * 2 * np.pi
        theta = 0

        return generate_shape(self.get_point, theta, theta_max, delta_theta)

    def get_point(self, theta):
        r = self.a + self.b * theta
        point = r * np.array([np.cos(theta), np.sin(theta)])
        return point


class Epitrochoid:
    def __init__(self, R, r, d, size):
        self.R = R
        self.r = r
        self.d = d
        self.rotations = denom((self.R + self.r) / self.r)
        self.size = size

    def get_point(self, t):
        return np.array([(self.R + self.r) * np.cos(t) - self.d * np.cos(((self.R + self.r) / self.r) * t),
                         (self.R + self.r) * np.sin(t) - self.d * np.sin(((self.R + self.r) / self.r) * t)]) * self.size

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 2 * np.pi * self.rotations

        return generate_shape(self.get_point, mini, maxi, step)


class GosperCurve:
    def __init__(self, size):
        self.size = size

    def make_shape(self, level):
        g = gosper.create_gosper_fractal(level)
        x, y = gosper.generate_level(g[level])
        shape = np.array(list(zip(x, y))) * self.size
        return shape


class EulerSpiral:
    def __init__(self, T, scale):
        self.T = T
        self.scale = scale

    def make_shape(self, N):
        shape = []

        t = 0
        n = N
        dt = self.T / N

        prev = np.zeros([2])
        shape.append(prev)
        while n > 0:
            dx = np.cos(t * t) * dt
            dy = np.sin(t * t) * dt
            t += dt

            point = prev + np.array([dx, dy]) * self.scale

            shape.append(point)

            prev = point
            n -= 1

        return np.vstack(shape)


class Polygon:
    def __init__(self, num, skip=1):
        self.num = num
        self.skip = skip

    def make_shape(self):
        shape = []
        for i in range(self.num + 1):
            a = 2 * np.pi / self.num * i * self.skip
            shape.append([np.cos(a), np.sin(a)])
        return np.array(shape)


class Wave:
    def __init__(self, waviness, scale):
        self.waviness = waviness
        self.scale = scale

        self.arc1 = circle_arc_3points(np.array([-2 * self.scale, 0]), np.array([-self.scale, waviness * scale]), np.array([0, 0]))
        self.arc2 = circle_arc_3points(np.array([0, 0]), np.array([self.scale, -waviness * scale]), np.array([2 * self.scale, 0]))

    def get_point(self, t):
        if t < 0.5:
            return self.arc1.get_point(t * 2 * self.arc1.length + self.arc1.angle)
        else:
            return self.arc2.get_point((t * 2 - 1) * self.arc2.length + self.arc2.angle)

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 1

        return generate_shape(self.get_point, mini, maxi, step)


class CircleArc:
    def __init__(self, middle, radius, angle, length):
        self.middle = middle
        self.radius = radius
        self.angle = angle
        self.length = length

    def get_point(self, t):
        return np.array([np.cos(t) * self.radius, np.sin(t) * self.radius]) + self.middle

    def make_shape(self, num):
        step = self.length / num
        mini = self.angle
        maxi = self.length + self.angle

        return generate_shape(self.get_point, mini, maxi, step)


def circle_arc_3points(p1, p2, p3):
    p1 = array_to_poi(p1)
    p2 = array_to_poi(p2)
    p3 = array_to_poi(p3)

    pc = circle_center(p1, p2, p3)
    r = pc.distance(p1)

    d1 = p1 - pc
    d2 = p2 - pc
    d3 = p3 - pc

    a1 = d1.getAngle()
    a2 = d2.getAngle()
    a3 = d3.getAngle()

    da1 = get_smallest_angle(a1, a2)
    da2 = get_smallest_angle(a2, a3)

    if da1 * da2 > 0:
        a2 = (da1 + da2) / 2 + a1
    else:
        a2 = (da1 + da2) / 2 + np.pi + a1

    da = get_smallest_angle(a1, a2) + get_smallest_angle(a2, a3)
    return CircleArc(pc.to_array(), r, a1, da)


class LineSegment:
    def __init__(self, start, delta):
        self.start = start
        self.delta = delta

    def get_point(self, t):
        return self.start + t * self.delta

    def make_shape(self, num):
        step = 1 / num
        mini = 0
        maxi = 1

        return generate_shape(self.get_point, mini, maxi, step)


def points_to_line_segment(p1, p2, norm=False):
    if norm:
        t = p2 - p1
        return LineSegment(p1, t / np.linalg.norm(t))
    return LineSegment(p1, p2 - p1)


def point_angle_to_line_segment(point, angle):
    return LineSegment(point, np.array([np.cos(angle), np.sin(angle)]))
