import torch
import torch.nn.functional as F
from torch import nn
import timm


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)
# v1
# class ImageDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder_layers = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU()
#         )
            
#     def forward(self, img):
#         # Input img: (B, 3, 96, 128)
#         # Output features: (B, 256, 12, 16)
#         return self.decoder_layers(img)

# class AimPointPredictor(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # This convolutional layer reduces feature channels to 1, creating a heatmap.
#         self.predictor_conv = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)
    
#     def forward(self, features):
#         # Input features: (B, 256, 12, 16)
#         heatmap = self.predictor_conv(features)  # Output heatmap: (B, 1, 12, 16)
#         # spatial_argmax expects input of size (BS, H, W)
#         # Squeeze the channel dimension (dim 1) before passing to spatial_argmax
#         return spatial_argmax(heatmap[:, 0]) # Output: (B, 2)

# class ImageDecoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.decoder_layers = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(384),
#             nn.ReLU(),
#             nn.Conv2d(384, 512, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU()
#         )
            
#     def forward(self, img):
#         # Input img: (B, 3, 96, 128)
#         # Output features: (B, 512, 24, 32)
#         return self.decoder_layers(img)

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1) # (B, C, H, W) -> (B, H, W, C)
        # Now, the 'C' dimension is the last one.
        # nn.LayerNorm(C) or super().forward(x) (if normalized_shape was set to C in __init__)
        # will normalize across this last dimension (channels) for each (B, H, W) location.
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
        return x

class ImageDecoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ImageDecoder, self).__init__()
        self.model = timm.create_model(
            'resnet18.fb_swsl_ig1b_ft_in1k',
            pretrained=pretrained,
            features_only=True,
        )
        self.conv = nn.Conv2d(512, 512, kernel_size=5, stride=2, padding=2)

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        output = self.model(x)
        output = F.interpolate(output[-1], size=(24, 32), mode='bilinear', align_corners=False)
        output = self.conv(output)
        return output


class AimPointPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # self.predictor_layers = nn.Sequential(
        #     nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
        #     LayerNorm2d(256),
        #     nn.ReLU(),
        #     nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
        #     LayerNorm2d(128),
        #     nn.ReLU(),
        #     nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        #     LayerNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
        #     LayerNorm2d(32),
        #     nn.ReLU(),
        # )
        self.heatmap_conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
    
    def forward(self, features):
        x = self.predictor_layers(features)
        heatmap = self.heatmap_conv(x)
        return heatmap[:, 0]
    

class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = ImageDecoder()
        self.predictor = AimPointPredictor()
        self._conv = torch.nn.Sequential(
            self.decoder,
            self.predictor
        )

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """
        x = self._conv(img)
        return spatial_argmax(x)


def save_model(model: Planner):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        save(model.decoder.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'decoder.th'))
        save(model.predictor.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'predictor.th'))
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from controller import control
    from utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=1000, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    test_planner(args)