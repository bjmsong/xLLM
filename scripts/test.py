import numpy as np

data=np.arange(24).reshape(2,3,4)
print(data)

dataT=data.transpose((1,0,2))
print(dataT)