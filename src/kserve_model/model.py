import kserve
from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image
import base64
import io


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 ):
        super(ConvBlock, self).__init__()
        to_pad = kernel_size // 2
        # чтобы не было артефактов на краях используем ReflectionPad2d
        self.reflection_pad = nn.ReflectionPad2d(to_pad)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(x))


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=channels, out_channels=channels, kernel_size=3, stride=1),
            nn.InstanceNorm2d(channels, affine=True),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 scale: int
                 ):
        super(UpsampleBlock, self).__init__()
        to_pad = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(to_pad)
        # чтобы не было шахматных артефактов, используем на выходе свертку
        self.upsample = nn.Upsample(scale_factor=scale, mode='nearest')
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        return self.conv(self.reflection_pad(self.upsample(x)))


class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()

        self.encoder = nn.Sequential(
            ConvBlock(in_channels=3, out_channels=32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(),
        )

        self.residual = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.decoder = nn.Sequential(
            UpsampleBlock(in_channels=128, out_channels=64, kernel_size=3, stride=1, scale=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(),
            UpsampleBlock(in_channels=64, out_channels=32, kernel_size=3, stride=1, scale=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(),
            ConvBlock(in_channels=32, out_channels=3, kernel_size=9, stride=1),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        residual = self.residual(encoded)
        decoded = self.decoder(residual)
        return decoded


class StyleModel(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.load()

    def load(self):
        model = torch.load(self.name, map_location=torch.device('cpu'))
        model.eval()
        self.model = model
        self.ready = True

    def predict(self, request: dict) -> dict:
        inputs = request["instances"]

        # Input follows the Tensorflow V1 HTTP API for binary values
        # https://www.tensorflow.org/tfx/serving/api_rest#encoding_binary_values
        data = inputs[0]["image"]["b64"]

        raw_img_data = base64.b64decode(data)
        input_image = Image.open(io.BytesIO(raw_img_data)).convert('RGB')

        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.mul(255)),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0).to('cpu')

        with torch.no_grad():
            output = self.model(input_batch).cpu()

        output = output.squeeze(0).clamp(0, 255).numpy()
        output = output.transpose(1, 2, 0).astype("uint8")
        result_image = Image.fromarray(output, 'RGB')

        byte_arr = io.BytesIO()
        result_image.save(byte_arr, format='jpg')
        encoded_img = base64.encodebytes(byte_arr.getvalue()).decode('ascii')

        return {"predictions": encoded_img}


if __name__ == "__main__":
    model = StyleModel("/mnt/models/model_wave.torch")
    model.load()
    kserve.ModelServer(workers=1).start([model])
