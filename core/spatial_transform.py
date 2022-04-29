import torch
import torch.nn.functional as F
import numpy as np


def _meshgrid(size):
    h, w = size
    x = torch.arange(0, w, dtype=torch.float32)
    y = torch.arange(0, h, dtype=torch.float32)
    z = torch.tensor(1, dtype=torch.float32)
    grid_x, grid_y,  grid_z = torch.meshgrid(x, y, z, indexing='ij')
    grid_xyz = torch.stack([grid_x, grid_y, grid_z], axis=-1)
    return grid_xyz.reshape(-1, 3)


def _pos_normalize(pos, size):
    pos = pos.clone()
    height, width = size
    pos[..., 0] = (pos[..., 0] / width - 0.5) * 2
    pos[..., 1] = (pos[..., 1] / height - 0.5) * 2
    return pos
    

def pt_spatial_transform_layer(feats, transform_matrix):
    '''Spatial transform layer.
    Args:
        feats: feature that needed to transform.
        transform_matrix: transform matrix between two coordinates,H@ P_1 = P_2, shape [N ,3, 3].
    Returns:
        warped features.
    '''
    N, C, H, W = feats.shape
    device = feats.device
    coord1_pos = _meshgrid((H, W))[None, ...].expand(N, -1, -1).to(device)
    coord2_pos = torch.bmm(coord1_pos, transform_matrix.permute(0, 2, 1))
    coord2_pos = coord2_pos[..., 0:2] /torch.maximum(
        coord2_pos[..., 2:3], torch.ones_like(coord2_pos[..., 2:3])*1e-5)

    # normalize to [-1, 1]
    coord2_pos = _pos_normalize(coord2_pos, (H, W))[..., :2].reshape(N, W, H, 2).permute(0, 2, 1, 3)
    
    # set outliers to 0
    mask1 = (coord2_pos[..., :1] >=-1) & (coord2_pos[..., :1] <=1)
    mask2 = (coord2_pos[..., 1:] >=-1) & (coord2_pos[..., 1:] <=1)
    mask = (mask1 & mask2).permute(0, 3, 1, 2).detach()
    
    warp_feats = F.grid_sample(feats, coord2_pos, align_corners=True)
    return warp_feats*mask


def two_images_warp(img1, img2, transform_matrix):
    '''Warp two images to best suitable coordinate.
    Args:
        img1, img2: two images, shape [3, H, W]
        transform_matrix: transform matrix between two coordinates, H@P_1 = P_2, shape [3, 3].
    Returns:
        new_img1, new_img2: Warped images.
        mask1, mask2: Valid maskes for two images.
    '''
    H1, W1 = img1.shape[1:]
    H2, W2 = img2.shape[1:]
    device = img1.device
    corners_coord2 = torch.tensor([[0., 0., 1.], [0., H2-1, 1.], [W2-1, 0., 1.], [W2-1, H2-1, 1.]]).to(device)
    corners_coord1 = torch.matmul(corners_coord2, torch.linalg.inv(transform_matrix).T)
    corners_coord1 /= corners_coord1[..., 2:3]

    (min_x, min_y), (max_x, max_y) = corners_coord1.min(dim=0).values[:2], corners_coord1.max(dim=0).values[:2]
    new_W = min(int(round(max(W1-1, max_x.item()) - min(0, min_x.item()))), 4*W1)
    new_H = min(int(round(max(W1-1, max_y.item()) - min(0, min_y.item()))), 4*H1)
    translate = torch.tensor([[1., 0., min(min_x, 0)],
                              [0., 1., min(min_y, 0)],
                              [0., 0., 1.]]).to(device)

    grid = _meshgrid((new_H, new_W)).to(device)
    # warp image1
    pos_img1 = torch.matmul(grid, translate.T)
    pos_img1[..., :2] /= pos_img1[..., 2:3]
    pos_img1_norm = _pos_normalize(pos_img1, (H1, W1))
    new_img1 = F.grid_sample(img1[None, ...], pos_img1_norm[..., :2].reshape(1, new_W, new_H, 2).permute(0, 2, 1, 3), align_corners=True)
    mask1 = F.grid_sample(torch.ones(1, 1, H1, W1).to(device), pos_img1_norm[..., :2].reshape(1, new_W, new_H, 2).permute(0, 2, 1, 3), align_corners=True)

    # warp image2
    pos_img2 = torch.matmul(pos_img1, transform_matrix.T)
    pos_img2[..., :2] /= pos_img2[..., 2:3]
    pos_img2_norm = _pos_normalize(pos_img2, (H2, W2))
    new_img2 = F.grid_sample(img2[None, ...], pos_img2_norm[..., :2].reshape(1, new_W, new_H, 2).permute(0, 2, 1, 3), align_corners=True)
    mask2 = F.grid_sample(torch.ones(1, 1, H2, W2).to(device), pos_img2_norm[..., :2].reshape(1, new_W, new_H, 2).permute(0, 2, 1, 3), align_corners=True)
    
    return new_img1, new_img2, mask1, mask2

def tf_spatial_transform_layer(image2_tensor, H_tf, raw=False):
    """Spatial Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        # Process
        # dim2 = width
        # dim1 = width*height
        # v = tf.range(num_batch)*dim1
        # print 'old v:', v # num_batch
        # print 'new v:', tf.reshape(v, (-1, 1)) # widthx1
        # n_repeats = 20
        # rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0]) # 1 x out_width*out_height
        # print rep
        # rep = tf.cast(rep, 'int32')
        # v = tf.matmul(tf.reshape(v, (-1, 1)), rep) # v: num_batch x (out_width*out_height)
        # print '--final v:\n', v.eval()
        # # v is the base. For parallel computing.
        #with tf.variable_scope('_repeat'):
            rep = torch.transpose(
                torch.unsqueeze(torch.ones([n_repeats,]), 1), 0, 1)
            rep = rep.type(torch.int32)

            x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
            return torch.reshape(x, [-1])
                
    def _interpolate(im, x, y, out_size):
        #with tf.variable_scope('_interpolate'):
            # constants
            num_batch = im.shape[0]
            height = im.shape[2]
            width = im.shape[3]
            channels = im.shape[1]

            x = x.type(torch.float32)
            y = y.type(torch.float32)
            height_f = float(height)
            width_f = float(width)
            out_height = out_size[0]
            out_width = out_size[1]

            max_y = (im.shape[2] - 1)
            max_x = (im.shape[3] - 1)



            #scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0) * (width_f) / 2.0
            y = (y + 1.0) * (height_f) / 2.0

            # do sampling
            x0 = torch.floor(x).type(torch.int32)
            x1 = x0 + 1
            y0 = torch.floor(y).type(torch.int32)
            y1 = y0 + 1


            x0 = torch.clamp(x0, min=0, max=max_x).to(im.device)      #将坐标划�?-127之间，超出部分用边界值表示，�?3�?表示
            x1 = torch.clamp(x1, min=0, max=max_x).to(im.device)
            y0 = torch.clamp(y0, min=0, max=max_y).to(im.device)
            y1 = torch.clamp(y1, min=0, max=max_y).to(im.device)
            dim2 = width
            dim1 = width * height
            base = _repeat(torch.arange(num_batch).type(torch.int32) * dim1, out_height * out_width).to(im.device)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = torch.reshape(im.permute(0, 2, 3, 1), [-1, channels])
            im_flat = im_flat.type(torch.float32)

            Ia = im_flat[idx_a.long()]
            Ib = im_flat[idx_b.long()]
            Ic = im_flat[idx_c.long()]
            Id = im_flat[idx_d.long()]

            # and finally calculate interpolated values
            x0_f = x0.type(torch.float32)
            x1_f = x1.type(torch.float32)
            y0_f = y0.type(torch.float32)
            y1_f = y1.type(torch.float32)
            wa = torch.unsqueeze(((x1_f - x) * (y1_f - y)), 1)
            wb = torch.unsqueeze(((x1_f - x) * (y - y0_f)), 1)
            wc = torch.unsqueeze(((x - x0_f) * (y1_f - y)), 1)
            wd = torch.unsqueeze(((x - x0_f) * (y - y0_f)), 1)
            output = wa * Ia + wb * Ib + wc * Ic + wd * Id
            return output
            #return Ia

    def _meshgrid(height, width):
        #with tf.variable_scope('_meshgrid'):


            x_t = torch.matmul(torch.ones([height, 1]),
                                torch.transpose(torch.unsqueeze(torch.linspace(0.0, 2.0, width) - 1.0, 1), 0, 1))
            y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, 2.0, height) - 1.0, 1),
                                torch.ones([1, width]))

            #print(x_t.eval())

            x_t_flat = torch.reshape(x_t, (1, -1))
            y_t_flat = torch.reshape(y_t, (1, -1))

            ones = torch.ones_like(x_t_flat)
            grid = torch.cat((x_t_flat, y_t_flat, ones), 0)
            # sess = tf.get_default_session()
            # print '--grid: \n', grid.eval() # (session=sess.as_default())
            return grid

    def _transform(image2_tensor, H_tf):
        #with tf.variable_scope('_transform'):
            num_batch = image2_tensor.shape[0]
            height = image2_tensor.shape[2]
            width = image2_tensor.shape[3]
            num_channels = image2_tensor.shape[1]
            #  Changed
            # theta = tf.reshape(theta, (-1, 2, 3))
            H_tf = torch.reshape(H_tf, (-1, 3, 3))
            H_tf = H_tf.type(torch.float32)

            #  Added: add two matrices M and B defined as follows in
            # order to perform the equation: H x M x [xs...;ys...;1s...] + H x [width/2...;height/2...;0...]
            H_tf_shape = list(H_tf.shape)
            # initial

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            out_height = height
            out_width = width
            grid = _meshgrid(out_height, out_width)
            grid = torch.unsqueeze(grid, 0)
            grid = torch.reshape(grid, [-1])
            grid = torch.tile(grid, (num_batch,))  # stack num_batch grids
            grid = torch.reshape(grid, [num_batch, 3, -1]).to(H_tf.device)

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = torch.matmul(H_tf, grid)
            x_s = T_g[:, 0:1, :]
            # x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            # Ty changed
            # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            y_s = T_g[:, 1:2, :]
            # y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            # Ty added
            # t_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])
            t_s = T_g[:, 2:3, :]
            # The problem may be here as a general homo does not preserve the parallelism
            # while an affine transformation preserves it.
            t_s_flat = torch.reshape(t_s, [-1])

            # # Avoid zero division
            # zero = tf.constant(0, dtype=tf.float32)
            # one = tf.constant(1, dtype=tf.float32)
            #
            # # smaller
            # small = tf.constant(1e-7, dtype=tf.float32)
            # smallers = 1e-6 * (one - tf.cast(tf.greater_equal(tf.abs(t_s_flat), small), tf.float32))
            #
            # t_s_flat = t_s_flat + smallers
            # condition = tf.reduce_sum(tf.cast(tf.greater(tf.abs(t_s_flat), small), tf.float32))

            #  batchsize * width * height
            x_s_flat = torch.reshape(x_s, [-1]) / t_s_flat
            y_s_flat = torch.reshape(y_s, [-1]) / t_s_flat

            input_transformed = _interpolate(image2_tensor, x_s_flat, y_s_flat, (height,width))

            output = torch.reshape(input_transformed, [num_batch, out_height, out_width, num_channels]).permute(0, 3, 1, 2)
            return output

    # with tf.variable_scope(name):
    batch_size = image2_tensor.shape[0]
    patch_size = float(image2_tensor.shape[3])
    M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                [0., patch_size / 2.0, patch_size / 2.0],
                [0., 0., 1.]]).astype(np.float32)
    M_tensor = torch.from_numpy(M)
    M_tile = torch.tile(torch.unsqueeze(M_tensor, 0), (batch_size, 1, 1)).to(H_tf.device)
    M_inv = np.linalg.inv(M)
    M_tensor_inv = torch.from_numpy(M_inv)
    M_tile_inv = torch.tile(torch.unsqueeze(M_tensor_inv, 0), (batch_size, 1, 1)).to(H_tf.device)
    H_1_mat = torch.matmul(torch.matmul(M_tile_inv, H_tf), M_tile)
    output = _transform(image2_tensor, H_1_mat)
    return output
