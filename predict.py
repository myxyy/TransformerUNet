from main import GPTUNet, model
import torchvision.transforms as transforms
import torch
import pytorch_lightning as pl
import numpy as np

np.set_printoptions(threshold=np.inf)

length_log_2 = 8
length = 2**length_log_2
vocab_size = 256
model.load_state_dict(torch.load('weight.pth'))
model = model.cuda()

prompt = '吾輩は'
prompt = torch.from_numpy(np.array([i for i in prompt.encode('utf-8')]).astype(np.int)).clone().cuda()
print(prompt)
prompt_len = len(prompt)
prompt = torch.nn.functional.pad(prompt, (0,length-prompt_len),'constant',0)

print(prompt.shape)

beam_width = 16
predict_init = model(prompt.view(1,length))
_, predict_init_i = predict_init.view(length, vocab_size)[prompt_len].topk(beam_width)
prompt_beam = prompt.repeat(beam_width, 1)
prompt_beam[:,prompt_len] = predict_init_i
prompt_len = prompt_len + 1

while prompt_len < length:
    predict_beam = model(prompt_beam)
    _, predict_beam_i = predict_beam[:,prompt_len,:].contiguous().view(beam_width * vocab_size).topk(beam_width)
    prompt_beam = prompt_beam[predict_beam_i // vocab_size]
    prompt_beam[:,prompt_len] = predict_beam_i % vocab_size 
    prompt_len = prompt_len + 1

predict = prompt_beam[0]
predict = predict.cpu().numpy().astype(dtype='uint8')
print(predict)
predict = predict.tobytes().decode('utf-8', 'replace')
print(predict)