import torch

path = "/home/haoyu/dog_up/Dreamwaq/logs/go2_up/Jan27_17-38-48_/model_21400.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=True)

sd = ckpt["model_state_dict"]
print("num tensors:", len(sd))

# 打印前 N 个参数
N = 50
for i, (k, v) in enumerate(sd.items()):
    if torch.is_tensor(v):
        print(f"{i:03d} {k:60s} {tuple(v.shape)} {v.dtype}  min={v.min().item():.4g} max={v.max().item():.4g}")
    else:
        print(f"{i:03d} {k}  type={type(v)}")
    if i + 1 >= N:
        break
