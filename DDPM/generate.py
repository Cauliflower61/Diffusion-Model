import torch
import DDPM
import torchvision


weight_path = './Checkpoint/epoch30.pkl'
batch = 1
in_c = 3
image_size = 64
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = DDPM.DDPM(in_c=in_c, timestep=300).to(device)
model.load_state_dict(torch.load(weight_path))

with torch.no_grad():
    imgs = model.sample(batch, in_c, image_size)
    for iter, img in enumerate(imgs):
        if iter == 0:
            x = img
        else:
            x = torch.concat([x, img], dim=0)
    torchvision.utils.save_image(x.cpu().data, f"./generate.png", nrow=20)