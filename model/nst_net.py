import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import torchvision.models as models


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach() # this is stage value (without compute gradient)
        self.loss = F.mse_loss(self.target, self.target) # just init

    def forward(self, inputs):
        self.loss = F.mse_loss(inputs, self.target)
        return inputs


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gramm_matrix(target_feature).detach() # this is stage value (without compute gradient)
        self.loss = F.mse_loss(self.target, self.target) # just init

    def forward(self, inputs):
        G = gramm_matrix(inputs)
        self.loss = F.mse_loss(G, self.target)
        return inputs


# ??????????????? batch_size f_maps_num width height
def gramm_matrix(inputs):
    batch_size, height, width, f_maps_num = inputs.size() # batch_size = 1
    features = inputs.view(batch_size * height, width * f_maps_num)
    G = torch.mm(features, features.t())
    return G.div(batch_size * height * width * f_maps_num)


cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, inputs):
        return (inputs - self.mean) / self.std


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

cnn = models.vgg19(pretrained=True).features.to(device).eval()


def get_style_model(cnn, norm_mean, norm_std, style_img, content_img,
                    content_layers=None, style_layers=None):
    if style_layers is None:
        style_layers = style_layers_default
    if content_layers is None:
        content_layers = content_layers_default
    cnn = copy.deepcopy(cnn)
    norm = Normalization(norm_mean, norm_std).to(device)
    content_losses = []
    style_losses = []

    model = nn.Sequential(norm)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv_{}'.format(i)
            i += 1
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i), content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module('style_loss_{}'.format(i), style_loss)
            style_losses.append(style_loss)

    # drop the layers after content and style layers because in deep layer more abstract
    # patterns for content, not for style
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]
    return model, style_losses, content_losses


def get_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


def train_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img,
                             num_iters=10, style_weight=10000, content_weight=1):
        model, style_losses, content_losses = get_style_model(cnn,
                                                              norm_mean, norm_std, style_img, content_img)
        optimizer = get_optimizer(input_img)

        print('Optimizing..')
        run = [0]
        while run[0] <= num_iters:

            def closure():
                # correct the values
                # это для того, чтобы значения тензора картинки не выходили за пределы [0;1]
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                # взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()

                return style_score + content_score

            optimizer.step(closure)

        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

