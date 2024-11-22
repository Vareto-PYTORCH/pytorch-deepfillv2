import numpy
import torch


def _init_conv_layer(conv, activation, mode='fan_out'):
    """
    Initializes the weights of a convolutional layer based on the activation function type.
    
    Args:
        conv (torch.nn.Conv2d): The convolutional layer to initialize.
        activation (torch.nn.Module): The activation function used in the layer (e.g., torch.nn.ReLU, torch.nn.LeakyReLU).
        mode (str): Initialization mode, typically 'fan_in' or 'fan_out'. Defaults to 'fan_out'.
    
    Applies Kaiming (He) initialization for the convolution weights:
    - For LeakyReLU activation, it uses the slope of the activation.
    - For ReLU and ELU, a specific 'relu' nonlinearity setting is used.
    - If the layer has a bias term, it initializes the bias to zero.
    """
    # Apply Kaiming uniform initialization with consideration of activation type
    if isinstance(activation, torch.nn.LeakyReLU):
        torch.nn.init.kaiming_uniform_(conv.weight, a=activation.negative_slope, nonlinearity='leaky_relu', mode=mode)
    elif isinstance(activation, (torch.nn.ReLU, torch.nn.ELU)):
        torch.nn.init.kaiming_uniform_(conv.weight, nonlinearity='relu', mode=mode)
    # Initialize bias to zero if it exists
    if conv.bias is not None:
        torch.nn.init.zeros_(conv.bias)


def output_to_image(out):
    """
    Converts a tensor output to an 8-bit RGB image format.
    
    Args:
        out (torch.Tensor): The output tensor with shape [batch, channels, height, width].
        
    Returns:
        numpy.ndarray: The output tensor as an 8-bit RGB image.
    """
    # Move tensor to CPU, permute dimensions to [height, width, channels], normalize to [0, 255]
    out = (out[0].cpu().permute(1, 2, 0) + 1.) * 127.5
    # Convert to uint8 format for image representation
    out = out.to(torch.uint8).numpy()
    return out


def same_padding(images, ksizes, strides, rates):
    """
    Applies 'SAME' padding similar to TensorFlow, ensuring the output size matches the input size.
    
    Args:
        images (torch.Tensor): Input tensor with shape [batch, channels, height, width].
        ksizes (tuple): Kernel sizes for height and width.
        strides (tuple): Stride sizes for height and width.
        rates (tuple): Dilation rates for height and width.
        
    Returns:
        torch.Tensor: Padded image tensor with padding applied to the sides.
    """
    in_height, in_width = images.shape[2:]
    
    # Calculate output dimensions based on input dimensions, strides, and padding mode
    out_height = -(in_height // -strides[0])  # ceil(in_height / strides[0])
    out_width = -(in_width // -strides[1])    # ceil(in_width / strides[1])
    
    # Calculate effective filter sizes with dilation (rate)
    filter_height = (ksizes[0] - 1) * rates[0] + 1
    filter_width = (ksizes[1] - 1) * rates[1] + 1
    
    # Determine padding amounts along height and width dimensions
    pad_along_height = max((out_height - 1) * strides[0] + filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * strides[1] + filter_width - in_width, 0)
    
    # Split padding into top/bottom and left/right for symmetry
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    
    # Apply zero-padding to the images
    paddings = (pad_left, pad_right, pad_top, pad_bottom)
    padded_images = torch.nn.ZeroPad2d(paddings)(images)
    
    return padded_images


def downsampling_nn_tf(images, n=2):
    """
    Performs nearest neighbor downsampling similar to TensorFlow's align_corners=True.
    
    Args:
        images (torch.Tensor): Input image tensor with shape [batch, channels, height, width].
        n (int): Downsampling factor. Defaults to 2.
        
    Returns:
        torch.Tensor: Downsampled image tensor.
    """
    in_height, in_width = images.shape[2:]
    
    # Calculate output dimensions based on downsampling factor
    out_height, out_width = in_height // n, in_width // n
    
    # Generate index tensors for height and width, aligning corners
    height_inds = torch.linspace(0, in_height - 1, steps=out_height, device=images.device).add_(0.5).floor_().long()
    width_inds = torch.linspace(0, in_width - 1, steps=out_width, device=images.device).add_(0.5).floor_().long()
    
    # Index and sample images based on the generated indices
    return images[:, :, height_inds][..., width_inds]


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extracts sliding local patches from input images using a sliding window.
    
    Args:
        images (Tensor): The input image tensor of shape [N, C, H, W].
        ksizes (list or tuple): Kernel size (height, width) for the patches.
        strides (list or tuple): Stride size (height, width) to extract the patches.
        rates (list or tuple): Dilation rates for the patches.
        padding (str): Padding mode ('same' or 'valid'). Default is 'same'.
    
    Returns:
        Tensor: Extracted patches from the images with shape [N, C*k*k, L], where L is the total number of patches.
    """
    # Apply padding to the images if 'same' padding is specified
    if padding == 'same':
        images = same_padding(images, ksizes, strides, rates)
        padding = 0

    # Use Unfold to extract sliding patches from the images
    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             stride=strides,
                             padding=padding,
                             dilation=rates)
    patches = unfold(images)  # [N, C*k*k, L]
    return patches  # Return extracted patches


def flow_to_image(flow):
    """
    Converts optical flow to color images for visualization.
    This function uses color coding for visualizing the motion field.

    Args:
        flow (Tensor): Optical flow tensor of shape [N, 2, H, W] where N is the batch size.
    
    Returns:
        numpy.ndarray: Color-encoded flow images.
    """
    out = []
    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.
    maxrad = -1
    
    # Iterate through each batch element
    for i in range(flow.shape[0]):
        u = flow[i, :, :, 0]  # Horizontal flow component
        v = flow[i, :, :, 1]  # Vertical flow component
        idxunknow = (abs(u) > 1e7) | (abs(v) > 1e7)  # Identify invalid flow
        u[idxunknow] = 0  # Set invalid flow components to 0
        v[idxunknow] = 0
        maxu = max(maxu, numpy.max(u))  # Track max and min values for normalization
        minu = min(minu, numpy.min(u))
        maxv = max(maxv, numpy.max(v))
        minv = min(minv, numpy.min(v))
        rad = numpy.sqrt(u ** 2 + v ** 2)  # Calculate magnitude of flow
        maxrad = max(maxrad, numpy.max(rad))
        
        # Normalize flow components u and v
        u = u / (maxrad + numpy.finfo(float).eps)
        v = v / (maxrad + numpy.finfo(float).eps)
        
        # Convert the flow components to a color image using a color map
        img = compute_color(u, v)
        out.append(img)  # Append each image to the output list
    
    return numpy.float32(numpy.uint8(out))  # Return the color-encoded flow images as uint8


def compute_color(u, v):
    """
    Converts flow components (u, v) into a color image using a color wheel.
    
    Args:
        u (numpy.ndarray): Horizontal flow component.
        v (numpy.ndarray): Vertical flow component.
    
    Returns:
        numpy.ndarray: Color-encoded image representing the flow direction and magnitude.
    """
    h, w = u.shape
    img = numpy.zeros([h, w, 3])  # Initialize the color image
    nanIdx = numpy.isnan(u) | numpy.isnan(v)  # Find NaN indices
    u[nanIdx] = 0
    v[nanIdx] = 0
    
    # Generate a color wheel for visualization
    colorwheel = make_color_wheel()
    ncols = numpy.size(colorwheel, 0)
    rad = numpy.sqrt(u ** 2 + v ** 2)  # Calculate flow magnitude
    a = numpy.arctan2(-v, -u) / numpy.pi  # Calculate flow direction
    fk = (a + 1) / 2 * (ncols - 1) + 1  # Map direction to color index
    k0 = numpy.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1  # Wrap around for color wheel
    f = fk - k0  # Fractional part for interpolation
    
    # Interpolate between the two closest colors
    for i in range(numpy.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1
        
        # Apply color based on flow magnitude
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = numpy.logical_not(idx)
        col[notidx] *= 0.75
        
        img[:, :, i] = numpy.uint8(numpy.floor(255 * col * (1 - nanIdx)))  # Store the color in the image
    
    return img  # Return the color image


def make_color_wheel():
    """
    Generates a color wheel for visualizing flow directions.
    
    Returns:
        numpy.ndarray: Color wheel with colors for different flow directions.
    """
    RY, YG, GC, CB, BM, MR = (15, 6, 4, 11, 13, 6)  # Number of colors in each segment of the wheel
    ncols = RY + YG + GC + CB + BM + MR  # Total number of colors
    colorwheel = numpy.zeros([ncols, 3])  # Initialize the color wheel
    
    col = 0
    # Red to Yellow (RY)
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = numpy.transpose(numpy.floor(255 * numpy.arange(0, RY) / RY))
    col += RY
    
    # Yellow to Green (YG)
    colorwheel[col:col + YG, 0] = 255 - numpy.transpose(numpy.floor(255 * numpy.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG
    
    # Green to Cyan (GC)
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = numpy.transpose(numpy.floor(255 * numpy.arange(0, GC) / GC))
    col += GC
    
    # Cyan to Blue (CB)
    colorwheel[col:col + CB, 1] = 255 - numpy.transpose(numpy.floor(255 * numpy.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB
    
    # Blue to Magenta (BM)
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = numpy.transpose(numpy.floor(255 * numpy.arange(0, BM) / BM))
    col += BM
    
    # Magenta to Red (MR)
    colorwheel[col:col + MR, 2] = 255 - numpy.transpose(numpy.floor(255 * numpy.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255
    
    return colorwheel  # Return the generated color wheel
    