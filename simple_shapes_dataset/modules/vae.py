import torch
from torch import nn


class RAEEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        ae_dim: int = 1028,
        z_dim: int = 16,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.dims = [
            ae_dim // (2**i) for i in reversed(range(4))
        ]  # 1 2 4 8 # 32 64 128 256

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.dims[-1] = ae_dim
        self.use_batchnorm = use_batchnorm

        self.out_dim = self.dims[3] * 2 * 2
        self.z_dim = z_dim

        self.layers = nn.Sequential(
            nn.Conv2d(
                num_channels,
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[0],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[1],
                self.dims[2],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.Conv2d(
                self.dims[2],
                self.dims[3],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[3])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.q_mean = nn.Linear(self.out_dim, self.z_dim)
        self.q_logvar = nn.Linear(self.out_dim, self.z_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.layers(x).view(x.size(0), -1)
        out = out.view(out.size(0), -1)

        return self.q_mean(out), self.q_logvar(out)


class RAEDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int,
        z_dim: int,
        ae_dim: int = 1028,
        kernel_dim: int = 4,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.dims = [ae_dim // (2**i) for i in reversed(range(3))]
        self.dims[-1] = ae_dim

        self.kernel_dim = kernel_dim
        self.padding = padding
        self.use_batchnorm = use_batchnorm

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim,
                self.dims[2],
                kernel_size=8,
                stride=1,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[2])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[2],
                self.dims[1],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[1])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
            nn.ConvTranspose2d(
                self.dims[1],
                self.dims[0],
                kernel_size=self.kernel_dim,
                stride=2,
                padding=self.padding,
                bias=not self.use_batchnorm,
            ),
            nn.BatchNorm2d(self.dims[0])
            if self.use_batchnorm
            else nn.Identity(),
            nn.ReLU(),
        )

        self.out_layer = nn.Sequential(
            nn.ZeroPad2d((0, 1, 0, 1)),
            nn.Conv2d(
                self.dims[0],
                self.num_channels,
                kernel_size=self.kernel_dim,
                stride=1,
                padding=self.padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out_layer(self.layers(z[:, :, None, None]))
