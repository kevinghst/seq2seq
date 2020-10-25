import numpy as np
import pdb
# with open('train.txt') as f:
#     content = f.readlines()
#
# train = content[:10000]
# test = content[10000:15000]
#
# with open('train_small.txt', 'w') as f:
#     for item in train:
#         f.write(item)
#
# with open('test_small.txt', 'w') as f:
#     for item in test:
#         f.write(item)


step_n = 10
steps = np.random.choice([-1, 0, 1], size=(1,2))
pdb.set_trace()
for n in range(step_n-1):
    new_step = np.random.choice([-1, 0, 1], size=(1,2))
    pdb.set_trace()
    steps = np.append(steps, new_step, axis=0)
    #something will be checked after each n

print(steps)