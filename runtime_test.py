from models import CCAStereo
import pandas as pd
from utils import efficiency_measure

print('')


kitti_shape= (1, 3, 384, 1248)
sceneflow_shape= (1, 3, 512, 960)

warmup = 100
iterations = 5000
stereo = True

models = [CCAStereo(192).cuda()]
modelnames = ['CCAStereo']

runtime_mean, runtime_std, total_ops, total_params = efficiency_measure(models,kitti_shape,iterations,warmup,stereo)

frame = pd.DataFrame(index=modelnames)
frame['latency mean (ms)'] = runtime_mean
frame['latency std'] = runtime_std
frame['macs (G)'] = total_ops
frame['parameters (M)'] = total_params


print('')
print('')
print(frame)
print('')
print('')