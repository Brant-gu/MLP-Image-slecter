import numpy as np

adv_X = np.load("adv_X.npy")
np.savetxt("adv_X.txt", adv_X, fmt="%.6f", delimiter=",")

adv_y = np.load("adv_y.npy")
np.savetxt("adv_y.txt", adv_y, fmt="%d")