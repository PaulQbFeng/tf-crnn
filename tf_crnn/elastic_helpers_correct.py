import tensorflow as tf


def sample(img, coords):
    """
    Args:
        img: bxhxwxc
        coords: bxh2xw2x2. each coordinate is (y, x) integer.
            Out of boundary coordinates will be clipped.
    Return:
        bxh2xw2xc image
    """
    shape = tf.shape(img)[1:3]   # h, w, c
    batch = tf.shape(img)[0]
    shape2 = tf.shape(coords)[1:3]  # h2, w2
    #assert None not in shape2, coords.get_shape()
    #max_coor = tf.cast(tf.stack([batch - 1, shape[0] - 1, shape[1] - 1, tf.shape(img)[3]]), tf.float32)
   # max_coor = tf.cast(tf.stack([shape[0] - 1, shape[1] - 1], tf.float32))

    # clip each dimension individually
    coords_y, coords_x = tf.split(coords, 2, axis=3)
    coords_y = tf.clip_by_value(coords_y, 0., tf.cast(
        shape[0] - 1, tf.float32))
    coords_x = tf.clip_by_value(coords_x, 0., tf.cast(
        shape[1] - 1, tf.float32))  
    coords = tf.concat([coords_y, coords_x], axis=3)


    #coords = tf.clip_by_value(coords, 0., tf.cast(
     #   shape[1] - 1, tf.float32))  # borderMode==repeat

    coords = tf.to_int32(coords)

    batch_index = tf.range(batch, dtype=tf.int32)
    batch_index = tf.reshape(batch_index, [-1, 1, 1, 1])
    batch_index = tf.tile(
        batch_index, [1, shape2[0], shape2[1], 1])    # bxh2xw2x1
    indices = tf.concat([batch_index, coords], axis=3)  # bxh2xw2x3

    print("indices", indices)
    sampled = tf.gather_nd(img, indices)
    print("sampled", sampled)
    return tf.cast(sampled, tf.float32)


def ImageSample(inputs, borderMode='repeat'):
    """
    Sample the images using the given coordinates, by bilinear interpolation.
    This was described in the paper:
    `Spatial Transformer Networks <http://arxiv.org/abs/1506.02025>`_.
    Args:
        inputs (list): [images, coords]. images has shape NHWC.
            coords has shape (N, H', W', 2), where each pair of the last dimension is a (y, x) real-value
            coordinate.
        borderMode: either "repeat" or "constant" (zero-filled)
    Returns:
        tf.Tensor: a tensor named ``output`` of shape (N, H', W', C).
    """
    image, mapping = inputs

    assert image.get_shape().ndims == 4 and mapping.get_shape().ndims == 4
    input_shape = tf.shape(image)[1:3]

    print("input_image_Sample", image.get_shape().as_list())
    print("input_mapping_Sample", mapping.get_shape().as_list())
    #assert None not in input_shape

    "Images in ImageSample layer must have fully-defined shape"
    assert borderMode in ['repeat', 'constant']

    orig_mapping = mapping
    mapping = tf.maximum(mapping, 0.0)
    lcoor = tf.floor(mapping)
    ucoor = lcoor + 1.0

    diff = mapping - lcoor
    neg_diff = 1.0 - diff  # bxh2xw2x2

    lcoory, lcoorx = tf.split(lcoor, 2, 3)
    ucoory, ucoorx = tf.split(ucoor, 2, 3)

    lyux = tf.concat([lcoory, ucoorx], 3)
    uylx = tf.concat([ucoory, lcoorx], 3)

    diffy, diffx = tf.split(diff, 2, 3)
    neg_diffy, neg_diffx = tf.split(neg_diff, 2, 3)

    ret = tf.add_n([sample(image, lcoor) * neg_diffx * neg_diffy,
                    sample(image, ucoor) * diffx * diffy,
                    sample(image, lyux) * neg_diffy * diffx,
                    sample(image, uylx) * diffy * neg_diffx], name='sampled')

    print("ret:", ret.get_shape().as_list())

    if borderMode == 'constant':
        max_coor = tf.cast(
            tf.stack([input_shape[0] - 1, input_shape[1] - 1]), tf.float32)
        mask = tf.greater_equal(orig_mapping, 0.0)
        mask2 = tf.less_equal(orig_mapping, max_coor)
        mask = tf.logical_and(mask, mask2)  # bxh2xw2x2
        mask = tf.reduce_all(mask, [3])  # bxh2xw2 boolean
        mask = tf.expand_dims(mask, 3)
        ret = ret * tf.cast(mask, tf.float32)

    return tf.identity(ret, name='output')


def _gauss_kernel(sigma, channels=1):
    """Return a gaussian kernel for the given sigma"""
    # compute length according to https://stackoverflow.com/a/25217058/786559

    size = 2 * tf.to_int32(4 * sigma + 0.5) + 1  # truncate at 4 std dev
    size = tf.squeeze(size)

    # implement according to https://stackoverflow.com/a/46526319/786559
    from_to = tf.to_float(tf.floor_div(size, 2))
    x = tf.linspace(-from_to, from_to, size)
    x /= sigma * tf.sqrt(2.)
    x2 = x**2
    kernel = tf.exp(- x2[:, None] - x2[None, :])
    kernel /= tf.reduce_sum(kernel)
    kernel = tf.stack([kernel] * channels, axis=2)
    kernel = tf.expand_dims(kernel, axis=-1)

    return kernel


def gaussian_filter_tf(image, sigma, name='Gaussian'):
    """Perform a 2D smoothing with a gaussian filter.

    This is a pure TensorFlow implementation of
    scipy.ndimage.gaussian_filter, with default values
    (truncate=4.0, order=0)

    Parameters:
        image: [W x H x C] tensor
        sigma: tensor of shape [1]

    NOTE: It uses 'SAME' padding for the convolution. However, as
    opposed to scipy, it does directly a 2D convolution, instead of
    a series of 1D convolutions for each axis
    """
    assert len(image.shape.as_list()
               ) == 3, 'Input image should be in WHC format'
    with tf.name_scope(name):
        with tf.name_scope('kernel'):
            kernel = _gauss_kernel(sigma, image.shape.as_list()[-1])
        image = tf.expand_dims(image, 0)  # add the batch component
        output = tf.nn.depthwise_conv2d(
            image, kernel, [1, 1, 1, 1], padding='SAME')
        #print("gaussian shape before squeeze", tf.shape(output).eval())
        #print("gaussian output:", output.get_shape().ndims)

    # Specify first and last dimension to be removed
    return tf.squeeze((output), [0, -1])


def tf_distortion_maps(img: tf.Tensor, batch_size: int = 128) -> tf.Tensor:

    # Input image (N,h,w,1)

    with tf.device("/device:GPU:0"):

        orig_shape = img.shape.as_list()  # output int32

        print("input shape:", img.get_shape().as_list())

        alpha = tf.cast(orig_shape[1], tf.float32)

        sigma = tf.abs(tf.random_normal([1], 8, 2))

        #sigma = tf.cond(sigma < 4 , lambda: 4 , lambda: sigma)

        # Output tensor of shape (h,w,1) with values between -1 and 1
        dispx = tf.random_uniform([orig_shape[1], orig_shape[2], 1], -1, 1)

        print("dispx", dispx.get_shape().as_list())

        dispy = tf.random_uniform([orig_shape[1], orig_shape[2], 1], -1, 1)

        dispx = alpha * gaussian_filter_tf(dispx, sigma)
        # TODO: make sure you use the same sigma ?
        dispy = alpha * gaussian_filter_tf(dispy, sigma)

    # use the broadcasting to achieve the same as meshgrid

        xs = tf.range(0, tf.cast(orig_shape[2], tf.float32), dtype=tf.float32)
        ys = tf.range(0, tf.cast(orig_shape[1], tf.float32), dtype=tf.float32)

        ys = tf.expand_dims(ys, axis=1)

        dispx += xs
        # print(tf.shape(xs).eval())
        dispy += ys
        # print(tf.shape(ys).eval())

        coords = tf.stack([dispy, dispx], axis=2)

        print("coords first stack ", coords.get_shape().as_list())

        #print("coords shape before expand :", tf.shape(coords).eval())
        # batch of 1
    #     coords = tf.expand_dims(coords, axis=0)
    #     print("coords shape after expand :", tf.shape(coords).eval())

        coords = [coords for i in range(batch_size)]

        coords = tf.stack(coords)  # stack coords to have dimension (B,H,W,2)

        print("coords stacked ", coords.get_shape().as_list())

        #print("coords shape final:" , tf.shape(coords).eval())

        # img = tf.expand_dims(img, axis=0)   #The image is dimension 3 (grayscale) so we add 1 dimension

    img = ImageSample((img, coords))
    print("dynamic image shape", img.get_shape().as_list())

    return img


def normalize_text(text):
    """Remove accents and other stuff from text"""
    return ''.join((c for c in unicodedata.normalize('NFD', text)
                    if unicodedata.category(c) != 'Mn'))
