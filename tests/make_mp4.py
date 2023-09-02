import os
import numpy as np
import imageio
from PIL import Image


FRAMES_PATH = 'datasets/2023-09-02'

def make_movie(frames, path, fps=15):
    path += '.mp4'
    writer = imageio.get_writer(path, format="mp4", mode="I", fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()


frames_filenames = sorted([filename for filename in os.listdir(FRAMES_PATH) if '.PNG' in filename])
frames_paths = [os.path.join(FRAMES_PATH, filename) for filename in frames_filenames]
frames = [np.array(Image.open(filepath)) for filepath in frames_paths]

make_movie(frames, os.path.join(FRAMES_PATH, 'video'))
