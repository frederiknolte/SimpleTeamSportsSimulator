import os
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm


FRAMES_PATHS = ['../tests/datasets/1_new',
                '../tests/datasets/1_old',
                ]
SAVE_PATH = '../tests/datasets/1.mp4'
FPS = 50


frames_paths = []
for path in FRAMES_PATHS:
    frames_filenames = sorted([filename for filename in os.listdir(path) if '.PNG' in filename])
    frames_paths.append([os.path.join(path, filename) for filename in frames_filenames])


example_frames = [Image.open(filepaths[0]) for filepaths in frames_paths]
max_len = max(*[len(l) for l in frames_paths])
writer = imageio.get_writer(SAVE_PATH, format="mp4", mode="I", fps=FPS)
for i in tqdm(range(max_len)):
    frame = []
    for j, dataset in enumerate(frames_paths):
        if i >= len(dataset):
            frame.append(np.zeros_like(example_frames[j]))
        else:
            frame.append(np.array(Image.open(dataset[i])))

    frame = np.concatenate(frame, axis=1)
    writer.append_data(frame)

writer.close()
