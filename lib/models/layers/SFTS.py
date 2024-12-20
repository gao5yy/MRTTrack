import torch
import torch.nn as nn
import math
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Cut & paste from PyTorch official master until it's in a few official releases - RW
# Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `model.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = model.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# Here, you can use the below function to get the Selection results in SUPP

def display_image(image_path, mode=1):
    pre_fix = '/13994058190/WYH/EDITOR/data/RGBNT201/train_171/'
    if mode == 1:
        pre_fix = pre_fix + 'RGB/'
    elif mode == 2:
        pre_fix = pre_fix + 'NI/'
    elif mode == 3:
        pre_fix = pre_fix + 'TI/'
    image = Image.open(pre_fix + image_path)
    resized_image = image.resize((128, 256))  # Resize to 256x128
    plt.imshow(resized_image)
    plt.axis('off')
    plt.show()


# Visualize the mask on the image
def visualize_multiple_masks(images, masks, mode, pre_fix, writer=None, epoch=None):
    num_images_to_display = 12  # Number of images to display
    images = images[:num_images_to_display]
    masks = masks[:num_images_to_display]
    num_rows = 2  # Number of rows in the display grid
    num_cols = 6  # Number of columns in the display grid
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 6))

    for i in range(num_images_to_display):
        # Reshape the mask to 16x8
        mask_2d = masks[i].reshape(16, 8).cpu().numpy()

        # Upscale the mask to 256x128
        mask_upscaled = np.kron(mask_2d, np.ones((16, 16)))

        # Append the appropriate mode prefix
        if mode == 1 or mode == 0 or mode == 4 or mode == 5:
            prefix = pre_fix + 'RGB/'
        elif mode == 2 or mode == 10:
            prefix = pre_fix + 'NI/'
        elif mode == 3 or mode == 11:
            prefix = pre_fix + 'TI/'

        # Load the original image
        image = Image.open(prefix + images[i])
        original_image = image.resize((128, 256))  # Resize to 256x128

        # Convert the image to numpy array
        original_np = np.array(original_image)

        # Apply a color to the mask (e.g., yellow)
        mask_color = np.array([0, 0, 0])  # Black color for the mask
        masked_image = np.where(mask_upscaled[..., None], original_np, mask_color)
        if mode == 0 or mode == 10 or mode == 11:
            masked_image = original_np
        row = i // num_cols
        col = i % num_cols

        # Display the masked image
        axes[row, col].imshow(masked_image)
        axes[row, col].axis('off')
    plt.tight_layout()
    plt.show()
    if writer is not None:
        if mode == 0:
            sign = 'Original'
        elif mode == 1:
            sign = 'RGB'
        elif mode == 2:
            sign = 'NIR'
        elif mode == 3:
            sign = 'TIR'
        elif mode == 4:
            sign = 'FRE'
        elif mode == 5:
            sign = 'ATTN'
        writer.add_figure('Person_Token_Select_' + sign, fig, global_step=epoch)


class Part_Attention(nn.Module):
    def __init__(self, ratio=0.5):
        super(Part_Attention, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        length = len(x)
        N = x[0].shape[2]
        B = x[0].shape[0]
        k = 0
        
        last_map = x[k]
        for i in range(k + 1, length):
            last_map = torch.matmul(x[i], last_map)
        last_map = last_map[:, :, 0, 1:]
        for i in range(last_map.shape[1]):
            _, topk_indices = torch.topk(last_map[:, i, :], int(N * self.ratio), dim=1)
            topk_indices = torch.sort(topk_indices, dim=1).values
            selected_tokens_mask = torch.zeros((B, N), dtype=torch.bool).cuda()
            selected_tokens_mask.scatter_(1, topk_indices, 1)
            if i == 0:
                max_index_set = selected_tokens_mask
            else:
                max_index_set = max_index_set | selected_tokens_mask

        return _, max_index_set


class SFTS(nn.Module):
    def __init__(self, ratio=0.5):
        super(SFTS, self).__init__()
        self.part_select = Part_Attention(ratio=ratio)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, RGB_attn, TIR_attn=None, img_path=None, writer=None, epoch=None):
        _, RGB_index = self.part_select(RGB_attn)
        _, TIR_index = self.part_select(TIR_attn)

        index = RGB_index | TIR_index
        result_true = torch.tensor([0, 1], dtype=torch.float32).cuda()
        result_false = torch.tensor([1, 0], dtype=torch.float32).cuda()

        mask = torch.where(index.unsqueeze(-1), result_true, result_false)
        return  mask
