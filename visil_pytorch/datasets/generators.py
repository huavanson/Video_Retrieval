import os
import glob
import utils
import torch
import numpy as np
import pickle
from torch.utils.data import Dataset


class VideoGenerator(Dataset):
    def __init__(self, video_file, fps=1, cc_size=224, rs_size=256):
        super(VideoGenerator, self).__init__()
        self.videos = np.loadtxt(video_file, dtype=str)
        self.videos = np.expand_dims(self.videos, axis=0) if self.videos.ndim == 1 else self.videos
        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):
        video = utils.load_video(self.videos[index][1], fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)
        return torch.from_numpy(video), self.videos[index][0]
    

class VideoGeneratorForInference(Dataset):
    def __init__(self, pickle_file):
        super(VideoGeneratorForInference, self).__init__()
        self.feature = pickle.load(open(pickle_file,'rb'))
        self.index_feature = self.__index_feature__()

    def __len__(self):
        return len(self.feature)

    def __index_feature__(self): 
        return {index:f for index,f in enumerate(self.feature)}

    def __getitem__(self, index):
        return self.feature[self.index_feature[index]], self.index_feature[index]


class DatasetGenerator(Dataset):
    def __init__(self, rootDir, videos, pattern, fps=1, cc_size=224, rs_size=256):
        super(DatasetGenerator, self).__init__()
        self.rootDir = rootDir
        self.videos = videos
        self.pattern = pattern
        self.fps = fps
        self.cc_size = cc_size
        self.rs_size = rs_size

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, idx):
        try:
            video = glob.glob(os.path.join(self.rootDir, self.pattern.replace('{id}', self.videos[idx])))
            video = utils.load_video(video[0], fps=self.fps, cc_size=self.cc_size, rs_size=self.rs_size)
            return torch.from_numpy(video), self.videos[idx]
        except:
            return torch.from_numpy(np.array([])), ''
        