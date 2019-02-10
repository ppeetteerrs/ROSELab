import numpy as np
from skimage import io, transform, util


def random_noise(im_data, im_size, target_size):
    return util.random_noise(im_data)


def random_crop(im_data, im_size, target_size):
    start_coords = np.random.randint(im_size - target_size + 1, size=(2,))
    return im_data[start_coords[0]:start_coords[0] + target_size, start_coords[1]:start_coords[1] + target_size]


def random_rotate(im_data, im_size, target_size):
    return transform.rotate(im_data, np.random.rand() * 360)


def random_flip(im_data, im_size, target_size):
    return np.fliplr(im_data)


# Data Augmentation
def augment(im_data, im_size, target_size):
    im_data = transform.resize(im_data, (im_size, im_size))
    im_data = random_noise(im_data, im_size, target_size)
    possible_augmentations = [random_rotate, random_flip]
    for augmentation in possible_augmentations:
        if np.random.rand() > 0.5:
            im_data = augmentation(im_data, im_size, target_size)
    im_data = random_crop(im_data, im_size, target_size)
    return im_data


def load_image(images: [], args):
    # Image Size
    im_size = args["im_size"]

    # Image Size to start with
    im_aug_size = int(im_size * 1.5)
    output = []
    # bar = Bar('Loading Images', max=len(images))
    for index, image in enumerate(images):
        try:
            # Read image
            image_data = io.imread(image["path"])
            image_data = augment(image_data, im_aug_size, im_size)
            label_vec = np.zeros(50, dtype=np.int)
            label_vec[image["label_index"]] = 1
            output.append({
                "image": image_data,
                "label": label_vec
            })
            # bar.next()
        except Exception as e:
            print('corrupted img', image["path"])

    if len(output) > 1:
        return output
    else:
        return output[0]
