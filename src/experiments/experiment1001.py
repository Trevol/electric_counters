import numpy as np

a = np.float32([1, 2, 3])
assert (a is np.float32(a))

a = np.float16([1, 2, 3])
assert (a is not np.float32(a))

a = [1, 2, 3]
assert (a is not np.float32(a))
#
# assert (np.asarray(a, np.float32).dtype == np.float32)




