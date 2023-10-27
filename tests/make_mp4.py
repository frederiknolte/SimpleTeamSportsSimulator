import os
import numpy as np
import imageio
from PIL import Image
from tqdm import tqdm


FRAMES_PATHS = ['PATH_TO_DATASET1',
                'PATH_TO_DATASET2']
SAVE_PATH = 'PATH_TO_SAVE'

def make_movie(frames, path, fps=15):
    print('Writing...')
    path += '.mp4'
    writer = imageio.get_writer(path, format="mp4", mode="I", fps=fps)
    for frame in tqdm(frames):
        writer.append_data(frame)
    writer.close()

frame_sets = []
print('Loading...')
for path in tqdm(FRAMES_PATHS):
    frames_filenames = sorted([filename for filename in os.listdir(path) if '.PNG' in filename])
    frames_paths = [os.path.join(path, filename) for filename in frames_filenames]
    frame_sets.append([np.array(Image.open(filepath)) for filepath in frames_paths])

# frame_sets = [s[:min(*[len(l) for l in frame_sets])] for s in frame_sets]
max_len = max(*[len(l) for l in frame_sets])
print('Processing...')
for frame_set in tqdm(frame_sets):
    frame_set.extend([np.ones_like(frame_set[0]) for _ in range(max_len - len(frame_set))])
frames = np.concatenate(frame_sets, axis=2)

make_movie(frame_sets, SAVE_PATH)
