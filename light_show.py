import board
import time
import neopixel
import numpy as np


def rainboic(pixels, num_iter):
    n = pixels.n

    def g(x):
        d2 = 2.0
        r1 = x
        r2 = x - (n - 1)
        r3 = x - 2*(n - 1)
        return 250 * (np.exp(-r1*r1/d2) + np.exp(-r2*r2/d2)
                      + np.exp(-r3*r3/d2))

    x0 = np.arange(n)
    dx = [0.0, 0.0, 0.0]
    for i in range(num_iter):
        c = [g(x0 + dx[w]) for w in range(3)]

        dx[0] += 0.1
        dx[1] += 0.15
        dx[2] += 0.2

        for w in range(3):
            if dx[w] > (n - 1):
                dx[w] -= (n - 1)

        for k in range(n):
            pixels[k] = (c[0][k], c[1][k], c[2][k])
        pixels.show()
        time.sleep(0.1)


def move(pixels, num_iter):
    N = pixels.n

    c = np.zeros((N, 3))
    for i in range(N):
        r = (i - N/2)
        c[i, :] = 255*np.exp(-r*r)

    vdt = 1.0
    Ix = np.eye((N))
    update = np.zeros_like(Ix)
    for i in range(N):
        update[i, i] -= vdt
        update[i, (i + 1) % N] += vdt

    for j in range(num_iter):
        c[:, 0] = (Ix + 0.98 * update) @ c[:, 0]
        c[:, 1] = (Ix + 0.96 * update) @ c[:, 1]
        c[:, 2] = (Ix + 0.94 * update) @ c[:, 2]
        for i in range(N):
            pixels[i] = c[i, :]
        pixels.show()
        time.sleep(0.1)


def joes_glow(pixels, num_iter):
    m = [1.0, 1.0]
    x = [0.0, 0.5]
    v = [0.2, 0.0]
    c = [(255, 0, 0), (0, 255, 0)]

    delta_t = 0.01
    tol = 0.01

    for n in range(num_iter):
        for i in range(2):
            x[i] += v[i] * delta_t
            if x[i] > 1.0:
                x[i] -= 1.0
            elif x[i] < 0:
                x[i] += 1.0

        if abs(x[0] - x[1]) < tol:
            m_t = m[0] + m[1]
            v_0 = (m[0] - m[1]) / m_t * v[0] + 2 * m[1] / m_t * v[1]
            v_1 = 2 * m[0] / m_t * v[0] + (m[1] - m[0]) / m_t * v[1]
            v[0] = v_0
            v[1] = v_1

        for k in range(pixels.n):
            pixels[k] = (0, 0, 0)
        for i in range(2):
            pixel = round(x[i] * pixels.n)
            if pixel == pixels.n:
                pixel = 0
            pixels[pixel] = c[i]
        pixels.show()
#        time.sleep(delta_t)


def warm_glow(pixels, num_iter, c0, c1):
    n = pixels.n
    t = np.zeros(n)
    c0 = np.array(c0)
    c1 = np.array(c1)
    for j in range(num_iter):
        for k in range(n):
            w = [np.sin(t[k])**2, np.cos(t[k])**2]
            pixels[k] = c0 * w[0] + c1 * w[1]
        t += 0.2*np.random.rand(n)
        pixels.show()
        time.sleep(0.05)


def three_way_glow(pixels, num_iter, c0, c1, c2):
    n = pixels.n
    t = np.zeros(n)
    zz = np.random.rand(n)*0.5
    c0 = np.array(c0)
    c1 = np.array(c1)
    c2 = np.array(c2)
    for j in range(num_iter):
        for k in range(n):
            w = [0.66*np.cos(t[k])**2,
                 0.66*np.cos(t[k] + 2*np.pi/3)**2,
                 0.66*np.cos(t[k] + 4*np.pi/3)**2]
            pixels[k] = c0 * w[0] + c1 * w[1] + c2 * w[2]
        t += zz*np.random.rand(n)
        pixels.show()
        time.sleep(0.05)


def glow(pixels, num_iter):
    n = pixels.n
    bright = 60
    w = np.zeros((n), dtype=np.int32)
    for j in range(num_iter):
        for i in range(0, 360, 3):
            th = np.radians(i)
            w = [bright * (1+np.cos(th + 2*k*np.pi/n)) for k in range(n)]
            for k in range(n):
                pixels[k] = (w[k], w[(k + 1) % n], w[(k + 2) % n])
            pixels.show()


def chase(pixels, num_iter):
    n = pixels.n
    for k in range(n):
        pixels[k] = (0, 0, 0)
    pixels.show()

    b = 120
    for j in range(num_iter):
        th = j / num_iter * 2 * np.pi
        r = np.array([b*(1 + np.cos(th)),
                      b*(1 + np.cos(th + 2 * np.pi / 3)),
                      b*(1 + np.cos(th + 4 * np.pi / 3))], dtype=int)

        for k in range(n):
            pixels[(k - 1) % n] = (0, 0, 0)
            pixels[k] = (r[0], r[1], r[2])
            pixels.show()
            time.sleep(0.1)
        for k in reversed(range(n)):
            pixels[(k + 1) % n] = (0, 0, 0)
            pixels[k] = (r[0], r[1], r[2])
            pixels.show()
            time.sleep(0.1)


def chase_accelerando(pixels, num_iter):
    n = pixels.n

    b = 120
    d = 0.2
    for j in range(num_iter):
        th = j / num_iter * 2 * np.pi
        r = np.array([b*(1 + np.cos(th)),
                      b*(1 + np.cos(th + 2 * np.pi / 3)),
                      b*(1 + np.cos(th + 4 * np.pi / 3))], dtype=int)

        for k in range(n):
            pixels[(k - 1) % n] = (0, 0, 0)
            pixels[k] = (r[0], r[1], r[2])
            pixels.show()
            time.sleep(d)
        for k in reversed(range(n)):
            pixels[(k + 1) % n] = (0, 0, 0)
            pixels[k] = (r[0], r[1], r[2])
            pixels.show()
            time.sleep(d)
        d *= 0.95


def prime_flash(pixels, num_iter):
    q = [31, 41, 29, 19, 23, 43, 97, 11]

    for j in range(num_iter):

        c = np.random.rand(3)*255

        for i, w in enumerate(q):
            if j % w == 0:
                pixels[i] = c
                pixels.show()
                pixels[i] = (0, 0, 0)
                pixels.show()

        time.sleep(0.01)


def gray_code(pixels, color1, color2):
    n = pixels.n

    for i in range(1 << n):
        gray = i ^ (i >> 1)

        for k in range(n):
            if gray & (1 << k) != 0:
                pixels[k] = color1
            else:
                pixels[k] = color2
        pixels.show()
        time.sleep(0.1)


N = 8
pixels = neopixel.NeoPixel(board.D18, 8, auto_write=False)


while True:
    rainboic(pixels, 1000)
    move(pixels, 500)

    warm_glow(pixels, 500, (200, 0, 0), (0, 200, 0))
    warm_glow(pixels, 500, (20, 220, 50), (50, 200, 200))
    warm_glow(pixels, 500, (120, 220, 50), (50, 200, 0))

    c1 = np.random.rand(3)*255
    c2 = np.random.rand(3)*255
    warm_glow(pixels, 500, c1, c2)

    three_way_glow(pixels, 500, (200, 0, 0), (200, 200, 0), (0, 200, 200))

    glow(pixels, 30)
