import numpy as np


class Poi:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def rotate(self, theta):
        x = np.cos(theta) * self.x - np.sin(theta) * self.y
        self.y = np.cos(theta) * self.y + np.sin(theta) * self.x
        self.x = x

    def __str__(self):
        return str(self.x) + ", " + str(self.y)

    def distance(self, point):
        return np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)

    def distance_squared(self, point):
        return (self.x - point.x) ** 2 + (self.y - point.y) ** 2

    def get_fraction_point(self, point, fraction):
        return (point - self) * fraction + self

    def __add__(self, point):
        return Poi(self.x + point.x, self.y + point.y)

    def __sub__(self, point):
        return Poi(self.x - point.x, self.y - point.y)

    def __mul__(self, i):
        return Poi(self.x * i, self.y * i)

    def __truediv__(self, i):
        return Poi(self.x / i, self.y / i)

    def __eq__(self, point):
        return self.x == point.x and self.y == point.y

    def __ne__(self, point):
        return self.x != point.x or self.y != point.y

    def mirror_point(self, line, dist):
        temp = -dist * (line.a * self.x + line.b * self.y - line.c) / (line.a ** 2 + line.b ** 2)
        x = temp * line.a + self.x
        y = temp * line.b + self.y
        return Poi(x, y)

    def getAngle(self):
        if self.y < 0:
            return -1 * np.arccos(self.x / self.distance(Poi(0, 0)))
        else:
            return np.arccos(self.x / self.distance(Poi(0, 0)))

    def rounded(self):
        return Poi(int(round(self.x)), int(round(self.y)))

    def to_string(self):
        return str(self.x), str(self.y)

    def to_array(self):
        return np.array([self.x, self.y])


def array_to_poi(a):
    return Poi(a[0], a[1])


class Line:
    def __init__(self, a=0, b=0, c=0):
        self.a = a
        self.b = b
        self.c = c

    def intersection(self, line2):
        a, b, c = self.a, self.b, self.c
        d, e, f = line2.a, line2.b, line2.c
        d1 = 1 / d
        if b == a * e * d1:
            return None
        y = (c - a * f * d1) / (b - a * e * d1)
        x = f * d1 - e * y * d1
        return Poi(x, y)

    def mirror_point(self, p, dist):
        temp = -dist * (self.a * p.x + self.b * p.y - self.c) / (self.a ** 2 + self.b ** 2)
        x = temp * self.a + p.x
        y = temp * self.b + p.y
        return Poi(x, y)


def points_to_line(p1, p2):
    if p1.x == p2.x:
        return Line(1, 0, p1.x)
    a = -1 * (p2.y - p1.y) / (p2.x - p1.x)
    c = p1.y + a * p1.x
    return Line(a, 1, c)


def point_angle_to_line(point, angle):
    if abs(angle) == 0.5 * np.pi:
        return Line(1, 0, point.x)
    a = -1 * np.tan(angle)
    c = point.y + a * point.x
    return Line(a, 1, c)


def circle_center(p1, p2, p3):
    t = p2.x * p2.x + p2.y * p2.y
    bc = (p1.x * p1.x + p1.y * p1.y - t) / 2.0
    cd = (t - p3.x * p3.x - p3.y * p3.y) / 2.0
    det = (p1.x - p2.x) * (p2.y - p3.y) - (p2.x - p3.x) * (p1.y - p2.y)

    det = 1 / det

    # Avoid NaN
    if det > 99:
        det = 99

    x = (bc * (p2.y - p3.y) - cd * (p1.y - p2.y)) * det
    y = ((p1.x - p2.x) * cd - (p2.x - p3.x) * bc) * det

    centre = Poi(x, y)
    return centre


def denom(d):
    for n in range(1, 100):
        if n * d % 1 == 0:
            return n
    return 0


def modulo(a, n):
    return a - np.floor(a / n) * n


def get_smallest_angle(a1, a2):
    return modulo((a2 - a1 + np.pi), (2 * np.pi)) - np.pi
