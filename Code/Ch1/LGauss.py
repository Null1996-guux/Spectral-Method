# find all zero points of Gauss-Legendre polynomial

import numpy as np
import sys

# evaluate GaussLegendrePolynomial
def l_gauss(N, m, x):
    L = np.zeros((N+1, m+1))
    if N == 0:
        return 1 if m == 0 else 0
    else:
        L[0, 0], L[1, 0] = 1, x
        if m == 0:
            for i in range(2, N+1):
                L[i, m] = \
                        ((2 * i - 1) * x * L[i-1, m] - (i - 1) * L[i-2, m]) / i
        else:
            L[0, 0], L[0, 1] = 1, 0
            L[1, 0], L[1, 1] = x, 1
            for i in range(2, N + 1):
                for j in range(0, i+1):
                    if j > m: break     # 打断, 否则当 m < N 时会超出数组索引
                    L[i, j] = \
                    ( j * (2 * i - 1) * L[i-1, j-1] \
                          + x * L[i-1, j] * (2 * i - 1) \
                          - (i - 1) * L[i-2, j] ) / i
    return L[-1, -1]

# evaluate GaussLegendrePolynomial
def find_zpts(N, m = 0, reseps = 1e-10):
    zpt_rec = np.zeros(N - m)
    find_count = 0
    H = N ** (-2)
    lbd = -1
    while find_count < N - m:
        rbd = lbd + H
        if l_gauss(N, m, lbd) * l_gauss(N, m, rbd) > 0: # 判断, 当前区间上没有零点时直接增加左短点并进入下次循环
            pass
        else: # 判断当前区间上有零点， 利用牛顿法在区间 (lbd, rbd) 上搜索零点, 
            x_cur = (lbd + rbd)/2 
            x_las = rbd
            while np.abs(x_cur - x_las) > reseps:
                x_las = x_cur
                x_cur -= l_gauss(N, m, x_cur) / l_gauss(N, m + 1, x_cur)
            zpt_rec[find_count] = x_cur
            find_count += 1
        lbd += H
    if find_count == N-m:
        print('successfully find %d zero points'%(N - m))
        return zpt_rec
    else:
        print('only find %d zero points'%(find_count))
        return zpt_rec

if __name__ == '__main__':
    N = int(sys.argv[1])
    m = int(sys.argv[2])
    reseps = eval(sys.argv[3])
    res = find_zpts(N, m, reseps)
    print(res)
