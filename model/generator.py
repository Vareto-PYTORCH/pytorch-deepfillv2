import torch

from .base_funcs import _init_conv_layer, downsampling_nn_tf, extract_image_patches, flow_to_image, output_to_image, same_padding


class GConv(torch.nn.Module):
    """
    Implements the gated 2D convolution layer introduced in 
    "Free-Form Image Inpainting with Gated Convolution" (Yu et al., 2019).
    
    This layer applies a gated mechanism to selectively control which parts 
    of the input are passed through, enhancing the model's ability to focus 
    on relevant features in tasks like image inpainting.

    Args:
        cnum_in (int): Number of input channels.
        cnum_out (int): Number of output channels.
        ksize (int): Kernel size for the convolution.
        stride (int): Stride of the convolution. Defaults to 1.
        rate (int): Dilation rate for dilated convolution. Defaults to 1.
        padding (str): Padding type. If 'same', applies "SAME" padding. Defaults to 'same'.
        activation (torch.nn.Module): Activation function to apply after convolution. Defaults to ELU.
    """

    def __init__(self, cnum_in, cnum_out, ksize, stride=1, rate=1, padding='same', activation=torch.nn.ELU()):
        super().__init__()
        
        self.activation = activation
        self.cnum_out = cnum_out
        
        # If using gated convolution, double output channels for gating mechanism,
        # unless it's a final RGB output layer or no activation is provided.
        num_conv_out = cnum_out if self.cnum_out == 3 or self.activation is None else 2 * cnum_out
        
        # Define the main convolution layer with no padding initially; padding will be applied dynamically.
        self.conv = torch.nn.Conv2d(cnum_in, num_conv_out, kernel_size=ksize, stride=stride, padding=0, dilation=rate)
        
        # Initialize the convolution layer weights based on the activation function
        _init_conv_layer(self.conv, activation=self.activation)
        
        # Store convolution parameters
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.padding = padding

    def forward(self, x):
        """
        Forward pass of the gated convolution layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: Output tensor after gated convolution and activation.
        """
        # Apply dynamic padding (similar to TensorFlow's "SAME" padding) if specified
        if self.padding == 'same':
            x = same_padding(x, [self.ksize, self.ksize], [self.stride, self.stride], [self.rate, self.rate])
        
        # Perform convolution
        x = self.conv(x)
        
        # If output channel count is 3 (e.g., RGB) or no activation, skip gating
        if self.cnum_out == 3 or self.activation is None:
            return x
        
        # Split the output into two parts: feature map (x) and gating mask (y)
        x, y = torch.split(x, self.cnum_out, dim=1)
        
        # Apply activation to feature map
        x = self.activation(x)
        
        # Apply sigmoid to gating mask to keep values between 0 and 1
        y = torch.sigmoid(y)
        
        # Element-wise multiply feature map by gating mask
        x = x * y
        return x


class GDeConv(torch.nn.Module):
    """
    Gated Deconvolution layer that performs upsampling followed by gated convolution.

    This layer increases the spatial resolution of the input feature map using nearest-neighbor
    interpolation, followed by a gated convolution to refine the upsampled features.

    Args:
        cnum_in (int): Number of input channels.
        cnum_out (int): Number of output channels.
        padding (int): Padding for the convolution in the GConv layer. Defaults to 1.
    """

    def __init__(self, cnum_in, cnum_out, padding=1):
        super().__init__()
        
        # Initialize a Gated Convolution layer with specified parameters
        self.conv = GConv(cnum_in, cnum_out, 3, stride=1, padding=padding)

    def forward(self, x):
        """
        Forward pass of the gated deconvolution layer.
        
        Args:
            x (torch.Tensor): Input tensor with shape [batch, channels, height, width].
        
        Returns:
            torch.Tensor: Upsampled and gated output tensor.
        """
        # Perform upsampling with nearest-neighbor interpolation (scale factor = 2)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest', recompute_scale_factor=False)
        
        # Pass the upsampled tensor through the gated convolution layer
        x = self.conv(x)
        return x
    

class ContextualAttention(torch.nn.Module):
    """
    Contextual Attention Layer Implementation.
    
    This layer computes contextual attention, allowing a model to focus on relevant regions
    of an image for tasks like image inpainting. The concept is introduced in 
    'Generative Image Inpainting with Contextual Attention' by Yu et al., 2019.
    
    Attributes:
        ksize (int): Kernel size for contextual attention.
        stride (int): Stride for extracting patches from background.
        rate (int): Dilation rate for matching.
        fuse_k (int): Kernel size for fusion step.
        softmax_scale (float): Scale factor for softmax attention.
        n_down (int): Number of downsampling operations.
        fuse (bool): Whether to apply fusion to improve attention accuracy.
        return_flow (bool): If True, return the optical flow for visualization.
        device_ids (list): List of device ids for distributed processing.
    """

    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10., n_down=2, fuse=True, return_flow=False, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.device_ids = device_ids
        self.n_down = n_down
        self.return_flow = return_flow


    def forward(self, f, b, mask=None):
        """
        Forward pass of the contextual attention layer.
        
        Args:
            f (tensor): Foreground features to be matched.
            b (tensor): Background features for matching.
            mask (tensor, optional): Binary mask indicating unavailable regions in background.
        
        Returns:
            y (tensor): Contextually-attended foreground feature.
            flow (tensor): Optical flow representing attention map (if return_flow is True).
        """
        device = f.device
        raw_int_fs, raw_int_bs = list(f.size()), list(b.size())  # Get the dimensions of f and b
        
        # Extract patches from background for matching
        kernel = 2 * self.rate
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel], strides=[self.rate*self.stride, self.rate*self.stride], rates=[1, 1], padding='same')
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1).permute(0, 4, 1, 2, 3)
        raw_w_groups = torch.split(raw_w, 1, dim=0)  # Split patches into groups

        # Downsample foreground and background
        f = downsampling_nn_tf(f, n=self.rate)
        b = downsampling_nn_tf(b, n=self.rate)
        int_fs, int_bs = list(f.size()), list(b.size())
        
        # Split foreground into groups
        f_groups = torch.split(f, 1, dim=0)

        # Extract smaller patches from downsampled background
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1).permute(0, 4, 1, 2, 3)
        w_groups = torch.split(w, 1, dim=0)  # Split smaller patches into groups

        # Prepare the mask, downsampling if necessary
        if mask is None:
            mask = torch.zeros([int_bs[0], 1, int_bs[2], int_bs[3]], device=device)
        else:
            mask = downsampling_nn_tf(mask, n=(2**self.n_down) * self.rate)
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize], strides=[self.stride, self.stride], rates=[1, 1], padding='same')
        m = m.view(mask.size(0), mask.size(1), self.ksize, self.ksize, -1).permute(0, 4, 1, 2, 3)[0]
        mm = (torch.mean(m, axis=[1, 2, 3], keepdim=True) == 0.).float().permute(1, 0, 2, 3)

        # Process each patch in foreground
        y= list()
        offsets= list()
        k = self.fuse_k
        scale = self.softmax_scale
        fuse_weight = torch.eye(k, device=device).view(1, 1, k, k)

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            # Normalize patches and compute attention scores
            wi = wi[0]
            max_wi = torch.sqrt(torch.sum(wi**2, dim=[1, 2, 3], keepdim=True)).clamp_min(1e-4)
            wi_normed = wi / max_wi

            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])
            yi = torch.nn.functional.conv2d(xi, wi_normed, stride=1)

            if self.fuse:
                # Fuse scores to encourage larger patch attention
                yi = yi.view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = torch.nn.functional.conv2d(yi, fuse_weight, stride=1).view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])
                yi = yi.permute(0, 2, 1, 4, 3).view(1, 1, int_bs[2]*int_bs[3], int_fs[2]*int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = torch.nn.functional.conv2d(yi, fuse_weight, stride=1).view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3)

            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3]) * mm
            yi = torch.nn.functional.softmax(yi * scale, dim=1) * mm

            # If returning optical flow, calculate offsets
            if self.return_flow:
                offset = torch.argmax(yi, dim=1, keepdim=True)
                if int_bs != int_fs:
                    times = (int_fs[2]*int_fs[3]) / (int_bs[2]*int_bs[3])
                    offset = ((offset + 1) * times - 1).to(torch.int64)
                offset = torch.cat([torch.div(offset, int_fs[3], rounding_mode='trunc'), offset % int_fs[3]], dim=1)
                offsets.append(offset)

            # Deconvolve for patch pasting
            yi = torch.nn.functional.conv_transpose2d(yi, raw_wi[0], stride=self.rate, padding=1) / 4.
            y.append(yi)

        y = torch.cat(y, dim=0).contiguous().view(raw_int_fs)

        if not self.return_flow:
            return y, None

        offsets = torch.cat(offsets, dim=0).view(int_fs[0], 2, *int_fs[2:])
        h_add = torch.arange(int_fs[2], device=device).view([1, 1, int_fs[2], 1]).expand(int_fs[0], -1, -1, int_fs[3])
        w_add = torch.arange(int_fs[3], device=device).view([1, 1, 1, int_fs[3]]).expand(int_fs[0], -1, int_fs[2], -1)
        offsets = offsets - torch.cat([h_add, w_add], dim=1)

        flow = torch.from_numpy(flow_to_image(offsets.permute(0, 2, 3, 1).cpu().numpy())) / 255.
        flow = flow.permute(0, 3, 1, 2)

        if self.rate != 1:
            flow = torch.nn.functional.interpolate(flow, scale_factor=self.rate, mode='bilinear', align_corners=True)

        return y, flow



class Generator(torch.nn.Module):
    """
    Generator network with a two-stage architecture used in image inpainting.
    
    Stage 1: Performs an initial inpainting of missing image parts.
    Stage 2: Refines the inpainting using contextual attention to match surrounding textures.
    
    Args:
        cnum_in (int): Number of input channels. Defaults to 5.
        cnum (int): Base channel number. Controls network width. Defaults to 48.
        return_flow (bool): If True, returns the offset flow from the contextual attention module. Defaults to False.
        checkpoint (str): Path to a pretrained model checkpoint. If provided, the model loads the weights from it.
    """
    
    def __init__(self, cnum_in=5, cnum=48, return_flow=False, checkpoint=None):
        super().__init__()
        
        # Stage 1: Initial layers with convolution and atrous branches
        # Convolutional Branch
        self.conv1 = GConv(cnum_in, cnum // 2, 5, 1, padding=2)
        self.conv2_downsample = GConv(cnum // 2, cnum, 3, 2)
        self.conv3 = GConv(cnum, cnum, 3, 1)
        self.conv4_downsample = GConv(cnum, 2 * cnum, 3, 2)
        self.conv5 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv6 = GConv(2 * cnum, 2 * cnum, 3, 1)

        # Atrous (dilated) convolutions to expand receptive fields in stage 1
        self.conv7_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=2, padding=2)
        self.conv8_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.conv9_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.conv10_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)
        
        # Upsampling layers for stage 1
        self.conv11 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv12 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.conv13_upsample = GDeConv(2 * cnum, cnum)
        self.conv14 = GConv(cnum, cnum, 3, 1)
        self.conv15_upsample = GDeConv(cnum, cnum // 2)
        self.conv16 = GConv(cnum // 2, cnum // 4, 3, 1)
        
        # Final output layer for stage 1
        self.conv17 = GConv(cnum // 4, 3, 3, 1, activation=None)
        self.tanh = torch.nn.Tanh()
        
        # Stage 2: Refinement network with convolution and attention branches
        # Convolutional Branch
        self.xconv1 = GConv(3, cnum // 2, 5, 1, padding=2)
        self.xconv2_downsample = GConv(cnum // 2, cnum // 2, 3, 2)
        self.xconv3 = GConv(cnum // 2, cnum, 3, 1)
        self.xconv4_downsample = GConv(cnum, cnum, 3, 2)
        self.xconv5 = GConv(cnum, 2 * cnum, 3, 1)
        self.xconv6 = GConv(2 * cnum, 2 * cnum, 3, 1)
        
        # Atrous convolutions for the convolutional branch
        self.xconv7_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=2, padding=2)
        self.xconv8_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=4, padding=4)
        self.xconv9_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=8, padding=8)
        self.xconv10_atrous = GConv(2 * cnum, 2 * cnum, 3, rate=16, padding=16)
        
        # Attention Branch
        self.pmconv1 = GConv(3, cnum // 2, 5, 1, padding=2)
        self.pmconv2_downsample = GConv(cnum // 2, cnum // 2, 3, 2)
        self.pmconv3 = GConv(cnum // 2, cnum, 3, 1)
        self.pmconv4_downsample = GConv(cnum, 2 * cnum, 3, 2)
        self.pmconv5 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.pmconv6 = GConv(2 * cnum, 2 * cnum, 3, 1, activation=torch.nn.ReLU())

        self.contextual_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10, fuse=False, device_ids=None, n_down=2, return_flow=return_flow)
        
        # Refinement layers after attention
        self.pmconv9 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.pmconv10 = GConv(2 * cnum, 2 * cnum, 3, 1)
        
        # Combined output layers
        self.allconv11 = GConv(4 * cnum, 2 * cnum, 3, 1)
        self.allconv12 = GConv(2 * cnum, 2 * cnum, 3, 1)
        self.allconv13_upsample = GDeConv(2 * cnum, cnum)
        self.allconv14 = GConv(cnum, cnum, 3, 1)
        self.allconv15_upsample = GDeConv(cnum, cnum // 2)
        self.allconv16 = GConv(cnum // 2, cnum // 4, 3, 1)
        
        # Final output layer for stage 2
        self.allconv17 = GConv(cnum // 4, 3, 3, 1, activation=None)
        
        self.return_flow = return_flow

        # Load checkpoint if provided
        if checkpoint is not None:
            generator_state_dict = torch.load(checkpoint, weights_only=True)['G']
            self.load_state_dict(generator_state_dict, strict=True)
        self.eval()


    def forward(self, x, mask):
        """
        Forward pass for the generator.
        
        Args:
            x (torch.Tensor): Input image tensor with shape [batch, channels, height, width].
            mask (torch.Tensor): Binary mask tensor where 1 indicates missing regions and 0 indicates known regions.
        
        Returns:
            tuple: Output tensors (x_stage1, x_stage2) from each stage. 
                   If return_flow is True, also returns the offset flow.
        """
        xin = x

        # Stage 1
        x = self.conv1(x)
        x = self.conv2_downsample(x)
        x = self.conv3(x)
        x = self.conv4_downsample(x)
        x = self.conv5(x)
        
        # Dilated convolutions for larger receptive field
        x = self.conv6(x)
        x = self.conv7_atrous(x)
        x = self.conv8_atrous(x)
        x = self.conv9_atrous(x)
        x = self.conv10_atrous(x)
        x = self.conv11(x)
        x = self.conv12(x)

        # Upsample back to original spatial dimensions
        x = self.conv13_upsample(x)
        x = self.conv14(x)
        x = self.conv15_upsample(x)
        x = self.conv16(x)
        
        # Final output layer and activation for stage 1
        x = self.conv17(x)
        x = self.tanh(x)
        x_stage1 = x

        # Stage 2: Refinement
        x = x * mask + xin[:, 0:3, :, :] * (1. - mask)

        # Convolutional Branch
        xnow = x
        x = self.xconv1(xnow)
        x = self.xconv2_downsample(x)
        x = self.xconv3(x)
        x = self.xconv4_downsample(x)
        x = self.xconv5(x)
        
        x = self.xconv6(x)
        x = self.xconv7_atrous(x)
        x = self.xconv8_atrous(x)
        x = self.xconv9_atrous(x)
        x = self.xconv10_atrous(x)
        x_hallu = x

        # Attention Branch
        x = self.pmconv1(xnow)
        x = self.pmconv2_downsample(x)
        x = self.pmconv3(x)
        x = self.pmconv4_downsample(x)
        x = self.pmconv5(x)
        
        x = self.pmconv6(x)
        x, offset_flow = self.contextual_attention(x, x, mask)
        x = self.pmconv9(x)
        x = self.pmconv10(x)
        pm = x
        x = torch.cat([x_hallu, pm], dim=1)

        # Final Refinement
        x = self.allconv11(x)
        x = self.allconv12(x)
        x = self.allconv13_upsample(x)
        x = self.allconv14(x)
        x = self.allconv15_upsample(x)
        x = self.allconv16(x)
        
        x = self.allconv17(x)
        x = self.tanh(x)
        x_stage2 = x

        if self.return_flow:
            return x_stage1, x_stage2, offset_flow
        
        return x_stage1, x_stage2


    @torch.inference_mode()
    def predict(self, image, mask, return_vals=['inpainted', 'stage1'], device='cuda'):
        """
        Performs inpainting inference on an input image with a mask, optionally returning various stages of the process.
        
        Args:
            image (torch.Tensor): Input image tensor with shape [C, H, W], where C is the channel count.
            mask (torch.Tensor): Binary mask tensor with shape [1, H, W], where masked areas are > 0.
            return_vals (list of str): List specifying which outputs to return. Options include 'inpainted', 'stage1', 'stage2', and 'flow'.
            device (str): Device on which to perform computation, e.g., 'cuda' or 'cpu'.

        Returns:
            list of torch.Tensor: List of requested output stages as specified in return_vals. Each output is an image tensor.
        """
        
        # Extract height and width of the input image
        _, h, w = image.shape
        grid = 8  # Define the grid size to adjust input dimensions for the network

        # Resize image and mask to be compatible with grid size and add batch dimension
        image = image[:3, :h // grid * grid, :w // grid * grid].unsqueeze(0)  # Keep only first 3 channels
        mask = mask[0:1, :h // grid * grid, :w // grid * grid].unsqueeze(0)  # Use only one channel for mask

        # Normalize the image to range [-1, 1]
        image = (image * 2 - 1.)

        # Convert mask to float where masked areas are 1 and unmasked areas are 0
        mask = (mask > 0.).to(dtype=torch.float32)

        # Apply mask to image, effectively removing masked areas
        image_masked = image * (1. - mask)

        # Prepare input for the inpainting network
        ones_x = torch.ones_like(image_masked)[:, 0:1, :, :]  # Create a single-channel tensor of ones
        x = torch.cat([image_masked, ones_x, ones_x * mask], dim=1)  # Concatenate image, ones, and masked areas

        # Perform the forward pass, possibly including flow offset if required
        if self.return_flow:
            x_stage1, x_stage2, offset_flow = self.forward(x, mask)
        else:
            x_stage1, x_stage2 = self.forward(x, mask)

        # Combine inpainted regions with the unmasked parts of the original image
        image_compl = image * (1. - mask) + x_stage2 * mask

        output = []  # Prepare list to hold requested outputs
        for return_val in return_vals:
            # Determine which stages to return based on return_vals
            if return_val.lower() == 'stage1':
                output.append(output_to_image(x_stage1))
            elif return_val.lower() == 'stage2':
                output.append(output_to_image(x_stage2))
            elif return_val.lower() == 'inpainted':
                output.append(output_to_image(image_compl))
            elif return_val.lower() == 'flow' and self.return_flow:
                output.append(offset_flow)
            else:
                print(f'Invalid return value: {return_val}')  # Handle any invalid options

        return output  # Return the list of requested outputs
