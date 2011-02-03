"""
Tentative implementation of RRCM from Vovk, Gammerman & Shafer's book. See README.

Probably buggy. Slow.
"""

import numpy
import scipy.linalg

def make_c_action(x, alpha):
    """
    x : shape (n, p) matrix, n objects with p attributes each
    alpha : regularisation weight

    returns function mapping vector w to (I - H)w,
    where H is the hat matrix for penalised least squares with
    a penalty of alpha ||c||^2 on the coeffs c. L2 norm.
    alpha assumed to be non-negative.
    """

    x = numpy.array(x) # ensure we have a copy
    (n, p) = x.shape
    xt = x.T
    xtx_reg = numpy.dot(xt, x) + alpha * numpy.eye(p)
    xtx_reg_factored = scipy.linalg.cho_factor(xtx_reg)

    def c_action(w_0):
        w_1 = numpy.dot(xt, w_0)
        w_2 = scipy.linalg.cho_solve(xtx_reg_factored, w_1)
        w_3 = numpy.dot(x, w_2)
        return w_0 - w_3

    return c_action

def synch_flip_on_b(a, b):
    """
    *in place modifications!*
    """
    mask = b < 0.0
    a[mask] *= -1
    b[mask] *= -1

def compute_intersection_points(a, b):
    """
    computes {y : |ai + bi * y| = |an + bn * y| for some i = 1, ..., n},

    ... ignoring the points only included by the special case where the entire real
    line is included... and also including -infty and infty
    """

    b_n = b[-1]
    a_n = a[-1]

    points = set()

    bneq_mask = b != b_n
    points.update((a_n - a[bneq_mask]) / (b[bneq_mask] - b_n))
    points.update((-a_n - a[bneq_mask]) / (b[bneq_mask] + b_n))

    if b_n != 0.0:
        beq_mask = numpy.logical_not(bneq_mask)
        aneq_mask = a != a_n
        mask = numpy.logical_and(beq_mask, aneq_mask)
        points.update((a[mask] + a_n) / (-2.0 * b[mask]))

    points.add(-numpy.inf)
    points.add(numpy.inf)

    return points

def get_interval_repr(lo, hi):
    if lo == - numpy.inf:
        return hi - 1.0
    elif hi == numpy.inf:
        return lo + 1.0
    else:
        return (lo + hi) / 2.0

def test_ineq(a_i, b_i, a_n, b_n, y):
    return numpy.abs(a_i + b_i * y) >= numpy.abs(a_n + b_n * y)

def compute_nj(a, b, points):
    nj = numpy.zeros(len(points), dtype = numpy.int)
    n = len(a)
    m = len(points)
    a_n = a[-1]
    b_n = b[-1]
    for i in xrange(n):
        for j in xrange(m - 1):
            # is it sufficient to test one point in interval?
            # lines can only intersect nontrivially at one
            # of the points, right? so we only need to test
            # one point y, right?
            y = get_interval_repr(points[j], points[j + 1])
            if test_ineq(a[i], b[i], a_n, b_n, y):
                nj[j] += 1
    return nj

def compute_mj(a, b, points):
    mj = numpy.zeros(len(points), dtype = numpy.int)
    n = len(a)
    m = len(points)
    a_n = a[-1]
    b_n = b[-1]
    for i in xrange(n):
        for j in xrange(1, m - 1):
            if test_ineq(a[i], b[i], a_n, b_n, points[j]):
                mj[j] += 1
    return mj

def confidence_region(nj, mj, n, points, epsilon):
    m = len(points)
    min_count = epsilon * n
    included_open_intervals = []
    included_points = []
    for j in xrange(m - 1):
        if nj[j] > min_count:
            included_open_intervals.append((points[j], points[j + 1]))
        if mj[j] > min_count:
            included_points.append(points[j])
    return included_open_intervals, included_points

def rrcm(x, y, alpha):
    c_action = make_c_action(x, alpha)
    a = c_action(numpy.hstack((y[:-1], (0.0,))))
    b = c_action(numpy.hstack((numpy.zeros(len(y) - 1), (1.0,))))

    synch_flip_on_b(a, b)

    points = compute_intersection_points(a, b)
    points = numpy.asarray(list(points))
    points.sort()

    nj = compute_nj(a, b, points)
    mj = compute_mj(a, b, points)

    return lambda epsilon : confidence_region(nj, mj, len(a), points, epsilon)

def main():

    n = 75
    p = 1
    alpha = 1.0

    x = numpy.random.uniform(-1.0, 1.0, (n, p))
    y = numpy.random.uniform(-1.0, 1.0, (n, )) * 0.6 - 2.1 * x[:, 0]

    gamma = rrcm(x, y, alpha)

    import pylab
    pylab.plot(x[:-1], y[:-1], 'ko')
    significance_regions = (
        (0.01, '#00ff00'),
        (0.1, '#ffff00'),
        (0.2, '#ff8800'),
        (0.5, '#ff0000'),
    )
    for epsilon, colour in significance_regions:
        open_intervals, points = gamma(epsilon)
        print open_intervals
        for (a, b) in open_intervals:
            style = '-'
            if a == -numpy.inf:
                a = min(y[:-1])
                style = '--'
            if b == numpy.inf:
                b = max(y[:-1])
                style = '--'
            pylab.plot([x[-1]] * 2, [a, b], style, color = colour, linewidth = 3)
        pylab.plot([x[-1]] * len(points), points, '+', color = colour,
                markeredgewidth = 0)
    pylab.title('Ridge Regression Confidence Machine')
    pylab.show()

if __name__ == '__main__':
    main()

