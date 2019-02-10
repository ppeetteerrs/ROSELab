import os
import logging
from random import shuffle


class DataSet(object):

    def __init__(self, dataset: [], transform_fn=None, args={}):
        self.data = dataset
        self.transform = transform_fn
        self.args = args

    def shuffle(self):
        shuffle(self.data)

    def __getitem__(self, index):
        if not isinstance(index, slice):
            if self.transform is None:
                return self.data[index]
            else:
                return self.transform([self.data[index]], self.args)
        else:
            if self.transform is None:
                return self.data[index]
            else:
                return self.transform(self.data[index], self.args)

    def __len__(self):
        return len(self.data)

class ImageLoader(object):

    def __init__(self, data_dir_path="Indian50Resized/", train_size=None, test_size=None, transform_fn=None,
                 im_size=60):
        self.train_size = train_size
        self.test_size = test_size
        self.transform_fn = transform_fn
        self.sample_size = 0;
        self.image_infos = []
        self.train_image_infos = []
        self.test_image_infos = []
        self.train_data = None
        self.test_data = None
        self.im_size = im_size
        self.data_dir_path = os.path.join(os.getcwd(), data_dir_path)
        self.all_labels = os.listdir(self.data_dir_path)
        self._get_all_image_infos()
        self._train_test_split()

    def _get_all_image_infos(self):
        image_infos = []
        for label_index, image_label in enumerate(self.all_labels):
            image_dir = os.path.join(self.data_dir_path, image_label)
            for image_name in os.listdir(image_dir):
                if image_name.endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(image_dir, image_name)
                    image_infos.append({
                        "label": image_label,
                        "dir": image_dir,
                        "name": image_name,
                        "path": image_path,
                        "label_index": label_index
                    })
        shuffle(image_infos)

        self.image_infos = image_infos
        self.sample_size = len(self.image_infos)
        logging.debug("{0} image infos parsed".format(self.sample_size))

    def reshuffle(self):
        self.train_data.shuffle()

    def _train_test_split(self):
        if self.train_size:
            if self.test_size:
                # Both train and test sizes provided
                if self.train_size + self.test_size > self.sample_size:
                    # Total size exceeded total sample count
                    train_test_ratio = int(self.train_size / self.test_size)
                    self.train_size = int(train_test_ratio * self.sample_size)
                    self.test_size = self.sample_size - self.train_size
                else:
                    self.test_size = 0
            else:
                # Only train size provided
                if self.train_size > self.sample_size:
                    self.train_size = self.sample_size
                    self.test_size = 0
        else:
            if self.test_size:
                # Only test size provided
                if self.test_size > self.sample_size:
                    self.test_size = self.sample_size
                    self.train_size = 0
                else:
                    self.train_size = self.sample_size - self.test_size
            else:
                # No size provided
                train_test_ratio = 4
                self.train_size = int(train_test_ratio * self.sample_size)
                self.test_size = self.sample_size - self.train_size
        self.train_data = DataSet(self.image_infos[:self.train_size], transform_fn=self.transform_fn, args={
            "im_size": self.im_size
        })
        self.test_data = DataSet(self.image_infos[self.train_size:], transform_fn=self.transform_fn, args={
            "im_size": self.im_size
        })
