import torch
import torch.nn.functional as F

def cost_volume(feat1, feat2, search_range):
    """A cost volume stores the data matching costs for associating a pixel from image1 
       with its corresponding pixels in image2.
    Args:
        feat1: Image1's feature
        feat2: Image2's feature
        search_range: Search range (maximum displacement), [-search_range, search_range]
    Returns:
        cost_vol: cost volume between two images, shape [N, (search_range*2+1)**2, H, W]
    """
    max_offset = search_range * 2 + 1
    padded_feat2 = F.pad(feat2, ( search_range, search_range, search_range, search_range))
    h, w = feat1.shape[2:]

    cost_vol = []
    for y in range(0, max_offset):
        for x in range(0, max_offset):
            cost = (feat1*padded_feat2[:, :, y:y+h, x:x+w]).mean(dim=1, keepdims=True)
            cost_vol.append(cost)
    cost_vol = torch.cat(cost_vol, dim=1)
    cost_vol = F.leaky_relu(cost_vol, negative_slope=0.1, inplace=True)
    return cost_vol

def fast_cost_volume(feat1, feat2, search_range):
    """Faster version of computing a cost volume."""
    n, c, h, w = feat1.shape
    block_feat2 = F.unfold(feat2, kernel_size=(h, w), padding=search_range).reshape(n, c, h, w, -1)
    cost_vol = (feat1[..., None] * block_feat2).mean(dim=1).permute(0, 3, 1, 2).contiguous()
    return cost_vol
    
    
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows-1)*strides[0]+effective_k_row-rows)
    padding_cols = max(0, (out_cols-1)*strides[1]+effective_k_col-cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images

# contextual correlation layer
def CCL(c1, warp):
    shape = list(warp.shape)
    batch_size = shape[0]
    channel = shape[1]
    h = shape[2]
    w = shape[2]

    kernel = 3
    stride = 1
    rate = 1
    padding = 1
    if shape[2] == 16:
        rate = 1
        stride = 1
        padding = 1
    elif shape[2] == 32:
        rate = 2
        stride = 1
        padding = 2
    else:
        rate = 3
        stride = 1
        padding = 3
    
    padded_warp = same_padding(warp, (kernel, kernel), (stride, stride), (rate, rate))
    unfold = torch.nn.Unfold(kernel_size=kernel, dilation=rate, padding=0, stride=stride)
    patches = unfold(padded_warp)
    patches = patches.view(batch_size, channel, kernel, kernel, h, w).permute(0, 4, 5, 1, 2, 3).contiguous().view(batch_size, -1, channel, kernel, kernel)

    # using convolution to match
    match_vol = []
    for i in range(batch_size):
        single_match = torch.nn.functional.conv2d(c1[i].unsqueeze(0), patches[i], padding=padding, dilation=rate)
        match_vol.append(single_match)

    match_vol = torch.cat(match_vol, 0)
    channels = int(match_vol.shape[1])
    
    # scale softmax
    softmax_scale = 10
    match_vol = torch.nn.functional.softmax(match_vol*softmax_scale, 1)

    # convert the correlation volume to feature flow
    h_one = torch.linspace(0., shape[2]-1., int(match_vol.shape[2]), device=match_vol.device)
    w_one = torch.linspace(0., shape[3]-1., int(match_vol.shape[3]), device=match_vol.device)
    h_one = torch.matmul(h_one.unsqueeze(1), torch.ones(1, shape[3], device=match_vol.device))
    w_one = torch.matmul(torch.ones(shape[2], 1, device=match_vol.device), torch.transpose(w_one.unsqueeze(1), 0, 1))
    h_one = torch.tile(h_one.unsqueeze(0).unsqueeze(0), (shape[0],channels,1,1))
    w_one = torch.tile(w_one.unsqueeze(0).unsqueeze(0), (shape[0],channels,1,1))
    
    i_one = torch.linspace(0., channels-1., channels).unsqueeze(0).to(match_vol.device)
    i_one = i_one.unsqueeze(-1).unsqueeze(-1)
    i_one = torch.tile(i_one, (shape[0], 1, shape[2], shape[3]))
 
    flow_w = match_vol*(i_one%shape[3] - w_one)
    flow_h = match_vol*(i_one//shape[3] - h_one)
    flow_w = flow_w.sum(dim=1, keepdim=True)
    flow_h = flow_h.sum(dim=1, keepdim=True)
    
    flow = torch.cat([flow_w, flow_h], 1)

    return flow


def fast_CCL(c1, warp):
    shape = list(warp.shape)
    batch_size = shape[0]
    channel = shape[1]
    h = shape[2]
    w = shape[2]

    kernel = 3
    stride = 1
    rate = 1
    padding = 1
    if shape[2] == 16:
        rate = 1
        stride = 1
        padding = 1
    elif shape[2] == 32:
        rate = 2
        stride = 1
        padding = 2
    else:
        rate = 3
        stride = 1
        padding = 3
    
    kernel_feat = F.unfold(warp, kernel_size=kernel, dilation=rate, padding=padding, stride=stride).permute(0, 2, 1)
    inputs_feat = F.unfold(c1, kernel_size=kernel, dilation=rate, padding=padding, stride=1)
    unfold_out = torch.bmm(kernel_feat, inputs_feat)
    match_vol = unfold_out.view(-1, h*w, h, w)

    channels = int(match_vol.shape[1])
    
    # scale softmax
    softmax_scale = 10
    match_vol = torch.nn.functional.softmax(match_vol*softmax_scale, 1)

    # convert the correlation volume to feature flow
    h_one = torch.linspace(0., shape[2]-1., int(match_vol.shape[2]), device=match_vol.device)
    w_one = torch.linspace(0., shape[3]-1., int(match_vol.shape[3]), device=match_vol.device)
    h_one = torch.matmul(h_one.unsqueeze(1), torch.ones(1, shape[3], device=match_vol.device))
    w_one = torch.matmul(torch.ones(shape[2], 1, device=match_vol.device), torch.transpose(w_one.unsqueeze(1), 0, 1))
    h_one = torch.tile(h_one.unsqueeze(0).unsqueeze(0), (shape[0],channels,1,1))
    w_one = torch.tile(w_one.unsqueeze(0).unsqueeze(0), (shape[0],channels,1,1))
    
    i_one = torch.linspace(0., channels-1., channels).unsqueeze(0).to(match_vol.device)
    i_one = i_one.unsqueeze(-1).unsqueeze(-1)
    i_one = torch.tile(i_one, (shape[0], 1, shape[2], shape[3]))
 
    flow_w = match_vol*(i_one%shape[3] - w_one)
    flow_h = match_vol*(i_one//shape[3] - h_one)
    flow_w = flow_w.sum(dim=1, keepdim=True)
    flow_h = flow_h.sum(dim=1, keepdim=True)
    
    flow = torch.cat([flow_w, flow_h], 1)

    return flow
