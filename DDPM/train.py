import torch
import datasets, DDPM
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter


# parameters setting #
Checkpoint_dir = './Checkpoint/'
log_dir = './TensorboardSave/'
CeleA_dir = "D:/neural network/Dataset/celebA/Img/img_align_celeba/"
MINST_dir = './datasets/MNIST_train.npy'
image_size = 64
batch_size = 64
epoch_num = 1000
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
timestep = 300
lr = 5e-4

# data loading #
train_datasets = datasets.CelebAData(CeleA_dir, image_size)
# train_datasets = datasets.MNISTData(file_npy=MINST_dir, image_size=image_size)
train_dataloader = DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
model = DDPM.DDPM(in_c=3, timestep=timestep).to(device)
optimizer = Adam(model.parameters(), lr=lr)
writer = SummaryWriter(log_dir)

model.train()
for epoch in range(epoch_num):
    total_loss = 0
    pbar = tqdm(total=len(train_dataloader), desc=f"epoch{epoch+1}/{epoch_num}")
    for iter, image in enumerate(train_dataloader):
        image = image.to(device)
        batch = image.shape[0]
        optimizer.zero_grad()
        t = torch.randint(0, timestep, (batch, ), device=device).long()
        noise, noisy = model.add_noise(image, t)
        predict = model(noisy, t)
        loss = model.loss(noise, predict)
        total_loss = total_loss + loss.item()
        loss.backward()
        optimizer.step()

        pbar.set_postfix(**{'Loss': loss.item()})
        pbar.update(1)
        global_step = epoch * len(train_dataloader) + iter
        writer.add_scalar(tag="training loss", scalar_value=loss.item(), global_step=global_step)

    pbar.close()
    print('total: %.4f' % (total_loss / len(train_dataloader)))
    # save model #
    torch.save(model.state_dict(), Checkpoint_dir + 'epoch' + str(epoch + 1) + '.pkl')
writer.close()