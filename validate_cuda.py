# validate_cuda.py
import torch
print('cuda_available:', torch.cuda.is_available())
print('device_count:', torch.cuda.device_count())
if torch.cuda.is_available():
    print('device0:', torch.cuda.get_device_name(0))
    x = torch.randn(1024,1024, device='cuda')
    y = torch.randn(1024,1024, device='cuda')
    z = torch.matmul(x,y)
    print('z[0,0]:', z[0,0].item())