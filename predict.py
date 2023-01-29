from main import GPTUNet
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np

np.set_printoptions(threshold=np.inf)

length_log_2 = 8
model = GPTUNet(length_log_2=length_log_2, depth_unet=length_log_2-1, depth_transformer=4, dim_scale=1.1)
model.load_state_dict(torch.load('weight.pth'))
model = model.cuda()

prompt = 'いつも'
prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(np.int)).clone().cuda()
print(prompt)
prompt_len = len(prompt)
prompt = torch.nn.functional.pad(prompt, (0,2**length_log_2-prompt_len),'constant',0)

beam_width = 16
prompt_beam = prompt.repeat(beam_width, 1)

while prompt_len < 2**length_log_2:
    predict_beam = model(prompt_beam)
    _, predict_beam_i = predict_beam[:,prompt_len,:].contiguous().view(beam_width * 256).topk(beam_width)
    prompt_beam[:,prompt_len][predict_beam_i // 256] = predict_beam_i % 256
    prompt_len = prompt_len + 1

predict = prompt_beam[0]
predict = predict.cpu().numpy().astype(dtype='uint8')
print(predict)
predict = predict.tobytes().decode('utf-8', 'replace')
print(predict)