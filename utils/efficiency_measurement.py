import torch
from tqdm import tqdm
import numpy as np
from thop import profile


def efficiency_measure(models,in_shape,iterations,warmup,stereo=False):

    input = torch.ones(in_shape).cuda()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = np.zeros((len(models),iterations)) 

    if stereo :
        input = (input,input)
    else:
        input = [input]

    with torch.no_grad():

        #Warmup
        for _ in range(warmup):
            for model in models:
                _ = model(*input)
    
        for iter in tqdm(range(iterations),total=iterations):
            for idx, model in enumerate(models):
                starter.record()
                _ = model(*input)
                ender.record()
                torch.cuda.synchronize()
                time = starter.elapsed_time(ender)
                timings[idx,iter] = time
        
        runtime_mean = np.round(np.mean(timings,axis=1),1)
        runtime_std = np.round(np.std(timings,axis=1),1)

    total_ops, total_params = profile(model,input,verbose=False)

    return runtime_mean, runtime_std, round(total_ops / (1000 ** 3),1), round(total_params / (1000 ** 2),1)