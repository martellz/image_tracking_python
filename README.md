# ferns-demo-python

convert ferns demo to python

the original codes are from [here](
    https://www.epfl.ch/labs/cvlab/software/descriptors-and-keypoints/ferns/
)

## Main changes

- use python3
- use opencv-python 4.x instead of opencv 1.x

## Environment

use python3.10 for example

```bash

conda create -n cv2 python=3.10

conda activate cv2

pip install opencv-python matplotlib tqdm

```

## sample data

download from http://pwp.ink/geshi.zip

## Run

see test.py for more details

```bash

python test.py [--image image_file] [--video video_file] [--tracker_file tracker_file] [--save_video save_video_file]

```

## TODO

- [x] tracker: implement tracking
- [x] detector: use orb detector instead of ferns detector in demo
- [ ] tracker: debug learning
- [ ] tracker: optimize tracking: vectorize, parallelize, trade off, etc.
- [ ] tracker: change failure strategy
- [ ] detector: implement ferns detector
- [ ] sdk: migrate to tfjs
- [ ] sdk: migrate to wasm
