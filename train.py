import argparse
import collections
import pathlib
import math

import torch
import torchvision


parser = argparse.ArgumentParser(description='Antoencoder with Adversarial Disentanglement')
parser.add_argument('--supervised', default=False, action='store_true')
parser.add_argument('--latent-split', default=[8, 4], nargs='+', type=int)
parser.add_argument('--beta', default=0.5, type=float)
parser.add_argument('--dis-channels', default=64, type=int)
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--manual-seed', type=int, help='manual seed')
parser.add_argument('--result-path', default='./result', help='folder to output images')
parser.add_argument('--dataset-path', required=True, help='folder to store dataset')

args = parser.parse_args()
print(args)

if args.manual_seed is not None:
    print("Random Seed: ", args.manual_seed)
    torch.manual_seed(args.manual_seed)

device = torch.device("cuda")

transform = torchvision.transforms.ToTensor()
dataset = torchvision.datasets.MNIST(args.dataset_path, True, transform, download=True)
loader = torch.utils.data.DataLoader(dataset, args.batch_size, True)


def _load_by_labels(data, targets, labels):
    indices = [torch.nonzero(targets == a) for a in labels]
    while True:
        chosen_indices = [i[torch.randint(0, len(i), (1, ))] for i in indices]
        chosen_indices = torch.cat(chosen_indices)
        yield data[chosen_indices] * (1.0 / 255)

sample_test = _load_by_labels(dataset.data, dataset.targets, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])


class Autoencoder(torch.nn.Module):
    def __init__(self, z_split):
        super().__init__()

        z_channels = sum(z_split)

        e_channels = 512
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(28 * 28, e_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(e_channels, e_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(e_channels, z_channels),
        )

        d_channels = 512
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(z_channels, d_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(d_channels, d_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(d_channels, 28 * 28),
            torch.nn.Sigmoid(),
        )

        self.dumper = Dumper(z_split)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        z = self.encoder(x)
        y = self.decoder(z)
        y = y.reshape(B, C, H, W)
        return y, z

    def dump(self, x):
        B, C, H, W = x.shape
        x = x.reshape(B, C * H * W)
        z = self.encoder(x)

        def d(z):
            BB, _ = z.shape
            y = self.decoder(z)
            y = y.reshape(BB, C, H, W)
            return y

        return {label: d(z_mixed) for (label, z_mixed) in self.dumper(z).items()}


class Dumper(torch.nn.Module):
    def __init__(self, z_split):
        super().__init__()

        n = len(z_split)
        entries = {}
        for i in range(n):
            mask = [1] * i + [0] + [1] * (n - 1 - i)
            if i == 0:
                label = 'dump'
                mask = [1 - m for m in mask]
            else:
                label = f'dump_{i}'
            mask_h = []
            for m, s in zip(mask, z_split):
                mask_h += [m] * s
            mask_h = torch.tensor(mask_h)
            mask_w = 1 - mask_h
            entry = torch.nn.Module()
            entry.register_buffer('mask_h', mask_h[None, :])
            entry.register_buffer('mask_w', mask_w[None, :])
            entries[label] = entry
        self.entries = torch.nn.ModuleDict(entries)

    def forward(self, z):
        def _mix(z, entry):
            B, C = z.shape
            zh = z * entry.mask_h
            zw = z * entry.mask_w
            zhzw = zh[:, None, :] + zw[None, :, :]
            zhzw = zhzw.reshape(B * B, C)
            return zhzw
        return {label: _mix(z, entry) for (label, entry) in self.entries.items()}


latent_split = args.latent_split
autoencoder = Autoencoder(latent_split).to(device)


class Discriminator(torch.nn.Module):
    def __init__(self, z_split, mid_channels):
        super().__init__()
        z_channels = sum(z_split)
        self.z_split = z_split
        self.net = torch.nn.Sequential(
            torch.nn.Linear(z_channels, mid_channels),
            torch.nn.Sigmoid(),
            torch.nn.Linear(mid_channels, 1, bias=False),
        )
        def _init_weight(m):
            if isinstance(m , torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')
        self.apply(_init_weight)

    def _shuffle(self, z):
        zz = torch.split(z, self.z_split, 1)
        zp = [torch.cat((zi[i:], zi[:i])) for (i, zi) in enumerate(zz)]
        return torch.cat(zp, 1)

    def forward(self, z, p):
        y_raw = self.net(z)
        y_shuffled = self.net(self._shuffle(z))

        y_relative = torch.mean(y_raw) - torch.mean(y_shuffled)
        loss = p * torch.nn.functional.softplus(-y_relative) + \
            (1 - p) * torch.nn.functional.softplus(y_relative)
        score = torch.sigmoid(y_relative)
        return loss, score


if args.supervised:
    dis_split = [10 + latent_split[0]] + latent_split[1:]
else:
    dis_split = latent_split
discriminator = Discriminator(dis_split, args.dis_channels).to(device)

optimizerD = torch.optim.Adam(discriminator.parameters())
optimizerG = torch.optim.Adam(autoencoder.parameters())


class Score():
    score_dict = collections.OrderedDict()

    def put(self, key, value):
        entry = self.score_dict.setdefault(key, [])
        entry.append(torch.mean(value.detach()))

    def print(self, epoch):
        print(f'{epoch}: ', end='')
        for (key, value) in self.score_dict.items():
            print(f'{key}={torch.mean(torch.stack(value)).item():.4f}, ', end='')
        print('')


def train(epoch):
    autoencoder.train()
    discriminator.train()
    score = Score()

    for (x, label) in loader:
        B, C, H, W = x.shape
        x = x.to(device)
        label = label.to(device)

        y, z = autoencoder(x)
        ae_loss = torch.nn.functional.binary_cross_entropy(y, x)
        if args.supervised:
            label_onehot = torch.eye(10, device=label.device)[label]
            zd = torch.cat((label_onehot, z), 1)
        else:
            zd = z
        gen_loss, _ = discriminator(zd, 0.0)
        ae_gen_loss = H * W * ae_loss + args.beta * gen_loss
        dis_loss, dis_score = discriminator(zd.detach(), 1.0)

        autoencoder.zero_grad(True)
        ae_gen_loss.backward()
        optimizerG.step()

        discriminator.zero_grad(True)
        dis_loss.backward()
        optimizerD.step()

        score.put('AE', ae_loss)
        score.put('Dis', dis_score)

        zz = torch.split(z, latent_split, 1)
        for i, zi in zip("abcdefghijklmnopqrstuvwxyz", zz):
            score.put(f'var(Z{i})', torch.var(zi))

    score.print(epoch)
        

def dump():
    autoencoder.train(False)

    x = next(sample_test)
    x = x.to(device)
    y, z = autoencoder(x)

    image = torch.stack((x, y), 1)
    B, T, C, H, W = image.shape
    image = image.reshape(B * T, C, H, W)
    torchvision.utils.save_image(image, result_path / 'autoencoder.png', nrow=T)

    generated = autoencoder.dump(x)
    for (label, dump) in generated.items():
        torchvision.utils.save_image(dump, result_path / f'{label}.png', nrow=B)


result_path = pathlib.Path(args.result_path)
result_path.mkdir(exist_ok=True)

for epoch in range(1, args.epochs + 1):
    train(epoch)
    dump()
