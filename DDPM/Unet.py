import torch
import torch.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_c, 2 * out_c)
        )

    def forward(self, t):
        t = self.mlp(t)
        return t


class Block(nn.Module):
    def __init__(self, in_c, out_c, embed_c, groups=32, embed_flag=False):
        super(Block, self).__init__()
        self.proj = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, out_c)
        self.act = nn.SiLU()
        self.embed_flag = embed_flag
        self.embed = TimeEmbedding(embed_c, out_c) if embed_flag else None

    def forward(self, x, t):
        x = self.proj(x)
        x = self.norm(x)
        if self.embed_flag:
            t = self.embed(t)[:, :, None, None]
            scale, shift = t.chunk(2, dim=1)
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, embed_c):
        super().__init__()
        self.block1 = Block(in_c, out_c, embed_c, embed_flag=True)
        self.block2 = Block(out_c, out_c, embed_c)
        self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0) if (in_c != out_c) else nn.Identity()

    def forward(self, x, t):
        x1 = self.block1(x, t)
        x1 = self.block2(x1, t)
        x2 = self.skip(x)
        return x1+x2


class DownSample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return self.conv(x)


class Unet(nn.Module):
    def __init__(self, in_c=3, base_c=64, embed_c=128, dim_block=(2, 4, 8, 16), timestep=1000):
        super().__init__()
        self.initial_conv = nn.Conv2d(in_c, base_c, kernel_size=1, padding=0)
        self.embed = nn.Sequential(
            nn.Embedding(timestep, embed_c),
            nn.GELU(),
            nn.Linear(embed_c, embed_c)
        )
        dims = [base_c * dim for dim in dim_block]
        self.encoder_block = nn.ModuleList([])
        self.encoder_block.append(ResidualBlock(base_c, base_c, embed_c))
        channels = base_c
        for i in range(len(dim_block)):
            self.encoder_block.append(
                nn.ModuleList([
                    DownSample(channels, channels),
                    ResidualBlock(channels, dims[i], embed_c),
                    ResidualBlock(dims[i], dims[i], embed_c),
                ])
            )
            channels = dims[i]

        dims = [*list(reversed(dims)), base_c]
        self.decoder_block = nn.ModuleList([])
        for i in range(len(dim_block)):
            if i == 0:
                self.decoder_block.append(UpSample(channels, dims[i+1]))
            else:
                self.decoder_block.append(
                    nn.ModuleList([
                        ResidualBlock(channels, dims[i], embed_c),
                        ResidualBlock(dims[i], dims[i], embed_c),
                        UpSample(dims[i], dims[i+1]),
                    ])
                )
            channels = dims[i]
        self.final_res = nn.ModuleList([
            ResidualBlock(channels, base_c, embed_c),
            ResidualBlock(base_c, base_c, embed_c),
            ])
        self.final = nn.Conv2d(base_c, in_c, kernel_size=1, padding=0)

    def forward(self, x, t):
        x = self.initial_conv(x)
        t = self.embed(t)
        down_list = []
        for iter, block in enumerate(self.encoder_block):
            if iter == 0:
                x = block(x, t)
            else:
                down, res1, res2 = block
                x = down(x)
                x = res1(x, t)
                x = res2(x, t)
            down_list.append(x)
        for iter, block in enumerate(self.decoder_block):
            if iter == 0:
                x = block(down_list.pop())
            else:
                res1, res2, up = block
                x = res1(x, t)
                x = res2(x, t)
                x = up(x)
            x = torch.concat((x, down_list.pop()), dim=1)
        for block in self.final_res:
            x = block(x, t)
        return self.final(x)


if __name__ == '__main__':
    model = Unet(in_c=1, base_c=64, embed_c=128, dim_block=[2, 4, 8], timestep=300)
    x = torch.randn(size=(16, 1, 32, 32))
    t = torch.randint(0, 300, (16, )).long()
    out = model(x, t)
    print(out.shape)
