import dynet as dy
import numpy as np

# This implementation will be unnecessarily slow, especially on the GPU.
# It can be improved by following the speed tricks covered in class:
# 1) Don't repeat operations.
# 2) Minimize the number of operations.
# 3) Minimize the number of CPU-GPU memory copies, make them earlier.

# Create the model
model = dy.ParameterCollection()
trainer = dy.SimpleSGDTrainer(model)
W_p = model.add_parameters((100,100))

# Create the "training data"
x_vecs = []
y_vecs = []
for i in range(10):
  x_vecs.append(np.random.rand(100))
  y_vecs.append(np.random.rand(100))

# Do the processing
for my_iter in range(1000):
  dy.renew_cg()
  W = dy.parameter(W_p)
  total = 0
  for x in x_vecs:
    for y in y_vecs:
      x_exp = dy.inputTensor(x)
      y_exp = dy.inputTensor(y)
      total = total + dy.dot_product(W * x_exp, y_exp)
  total.forward()
  total.backward()
  trainer.update()

