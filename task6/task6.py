import numpy as np
import json

def task(s: str) -> list:
    df = np.array(json.loads(s))

    r = []
    for i in range(df.shape[1]):
        r_i = []
        for j in range(df.shape[0]):
            r_ij = []
            for k in range(df.shape[0]):
                if df[j][i] == df[k][i]:
                    r_ij.append(0.5)
                elif df[j][i] > df[k][i]:
                    r_ij.append(0)
                else:
                    r_ij.append(1)
            r_i.append(r_ij)
        r.append(r_i)

    r = np.array(r)

    x = r.mean(0)

    k = np.array([1/3, 1/3, 1/3])
    k_old = k.copy()

    E = 0.001
    while np.linalg.norm(k - k_old) < E:
        k_old = k.copy()
        y = x @ k
        lmd = np.array([1.0, 1.0, 1.0]) @ (x @ k)
        k = (1 / lmd) * y

    return k.tolist()