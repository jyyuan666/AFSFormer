from torchvision.models.segmentation import fcn_resnet50
import torch.nn as nn
import torch


class FCN(nn.Module):
    def __init__(self, n_class=2):

        super(FCN, self).__init__()
        self.fcn = fcn_resnet50(pretrained=False, num_classes=2)

    def forward(self, x, debug=False):
        return self.fcn(x)['out']

    def resume(self, file, test=False):
        import torch
        if test and not file:
            self.fcn = fcn_resnet50(pretrained=True, num_classes=2)
            return
        if file:
            print('Loading checkpoint from: ' + file)
            checkpoint = torch.load(file)
            checkpoint = checkpoint['model_state_dict']
            self.load_state_dict(checkpoint)

model = FCN(2)

import numpy as np
device = torch.device('cuda')
model.to(device)
dummy_input = torch.randn(1, 3, 1024, 1024,dtype=torch.float).to(device)
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
repetitions = 300
timings=np.zeros((repetitions,1))
#GPU-WARM-UP
for _ in range(10):
   _ = model(dummy_input)
# MEASURE PERFORMANCE
with torch.no_grad():
  for rep in range(repetitions):
     starter.record()
     _ = model(dummy_input)
     ender.record()
     # WAIT FOR GPU SYNC
     torch.cuda.synchronize()
     curr_time = starter.elapsed_time(ender)
     timings[rep] = curr_time
mean_syn = np.sum(timings) / repetitions
std_syn = np.std(timings)
mean_fps = 1000. / mean_syn
print(' * Mean@1 {mean_syn:.3f}ms Std@5 {std_syn:.3f}ms FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
print(mean_syn)

