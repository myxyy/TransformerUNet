from main import GPTUNet
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np

np.set_printoptions(threshold=np.inf)

length_log_2 = 9
model = GPTUNet(length_log_2=length_log_2, depth_unet=length_log_2-1, depth_transformer=4, dim_scale=1.1)
model.load_state_dict(torch.load('weight.pth'))
model = model.cuda()

prompt = '吾輩は猫である。\n'
prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(np.int)).clone().cuda()
print(prompt)
prompt_len = len(prompt)
prompt = torch.nn.functional.pad(prompt, (0,1024-prompt_len),'constant',0)

while prompt_len < 1024:
    predict = model(prompt)
    p = predict[prompt_len]
    prompt[prompt_len] = p
    prompt_len = prompt_len + 1

predict = prompt.cpu().numpy().astype(dtype='uint8')
print(predict)
predict = predict.tobytes().decode('utf-8', 'replace')
print(predict)