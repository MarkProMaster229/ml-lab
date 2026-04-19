block_hash = "00000000000000000001b908e4f2c9cce9985181b3f6111a9a92373752b2e4d5"

seed = int(block_hash[-8:], 16)
print(seed)

import random
random.seed(seed)


import random
import string


alphabet = string.ascii_letters

test_samples = [random.choice(alphabet) for _ in range(15)]
print(test_samples)