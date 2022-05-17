import numpy as np


def Task1():
    print("Task1")

    N = 6                                       # const, my variant
    DOTS_AMOUNT = max(20-N, N) - (N % 3) - 2    # const
    M = DOTS_AMOUNT-1                           # const, size of polynom

    def test_func(x):
        return N*pow(x, max(20-N, N)) + (N-1)*pow(x, max(20-N, N) - 3) + (N+1)*pow(x, max(20-N, N) - 5)

    # def target_func(x, y, a):
    #     S = 0
    #     for i in range(0, DOTS_AMOUNT-1):
    #         tmp = 0
    #         for j in range(0, M-1):
    #             tmp += a[j]*pow(x[i], j)
    #         tmp -= y[i]
    #         S += pow(tmp, 2)
    #     return S

    x = np.zeros(DOTS_AMOUNT)               # inputs
    y = np.zeros(DOTS_AMOUNT)               # outputs

    for i in range(0, DOTS_AMOUNT):
        x[i] = i/(DOTS_AMOUNT-1)
        y[i] = test_func(x[i])

    C = np.zeros(2*M+1)                     # values for coefficient matrix
    D = np.zeros(M+1)                       # free-member vector

    for i in range(0, DOTS_AMOUNT):
        z = x[i]
        F = y[i]                            # func value
        E = 1                               # pow of z
        for j in range(0, 2*M+1):
            C[j] += E
            if j <= M:
                D[j] += F*E
            E *= z

    B = np.zeros((M+1, M+1))                # coefficient matrix
    a = np.zeros(M)                         # coefficients of polynom

    for i in range(0, M+1):
        for j in range(0, M+1):
            B[i][j] = C[i+j]

    a = np.linalg.solve(B, D)

    x_test = np.full(M+1, 1)

    answer = 'Polynom: '
    for i in range(0, M+1):
        answer += str(round(a[i], 3)) + '*x^' + str(i)
        if i != M:
            answer += ' + '
    print(answer)
    print()

    print('Supposed y if x=1: ' + str(round(a.dot(x_test), 3)))
    print('True value: ' + str(test_func(1)))

####################################################################


def Task2():
    print("Task2")

    N = 6                                       # const, amount of nodes
    START_X, LAST_X = 3, 8                      # const, range
    H = (LAST_X-START_X)/(N-1)                  # step

    def test_func(x):
        return 2*pow(x, 7) + 3*pow(x, 6) + 3*pow(x, 4) - 3

    x = np.zeros(N)                             # inputs
    y = np.zeros(N)                             # outputs

    for node in range(START_X, LAST_X+1):
        x[node - START_X] = node
        y[node - START_X] = test_func(node)

    A = np.zeros((N-2, N-2))                        # matrix to find mi
    np.fill_diagonal(A[1:], H/6)
    np.fill_diagonal(A[:, 1:], H/6)
    np.fill_diagonal(A, 2*H/3)

    b = np.zeros(N-2)                               # vector to find mi

    for i in range(2, N):
        b[i-2] = (y[i] + y[i-2] - 2*y[i-1])/H

    m = np.zeros(N)
    m[1:N-1] = np.linalg.solve(A, b)

    c = np.zeros((N-1, 4))

    for i in range(1, N):
        alpha = 6*y[i-1] - m[i-1]*pow(H, 2)
        beta = 6*y[i] - m[i]*pow(H, 2)

        delta = m[i-1]
        gamma = m[i]
        for j in reversed(range(0, 4)):
            c[i-1][j] = (pow(-1, j+1) * gamma + pow(-1, j) * delta) / 6*H
            if j in (1, 2):
                c[i-1][j] *= 3
            if j <= 1:
                c[i-1][j] += (pow(-1, j+1) * beta + pow(-1, j) * alpha) / 6*H
                alpha *= x[i]
                beta *= x[i-1]
            delta *= x[i]
            gamma *= x[i-1]

    x_test = np.array([1, 3, 9, 27])

    for i in range(0, N-1):
        answer = 'Polynom for range [' + str(START_X + i*H) + ';' + str(START_X + (i+1)*H) + ']:\n'
        for j in range(0, 4):
            answer += str(round(c[i][j], 3)) + '*x^' + str(j)
            if j != 3:
                answer += ' + '
        print(answer)

    print()
    print("Supposed y if x=3: " + str(round(c[0].dot(x_test), 3)))
    print("True value: " + str(test_func(3)))
    return


Task1()
print()
Task2()
