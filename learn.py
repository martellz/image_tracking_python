import argparse
import cv2
from src.tracker.template_matching_based_tracker import Template_matching_based_tracker

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='model.bmp')
    parser.add_argument('--tracker_file', type=str, default='temp.npy')
    parser.add_argument('--tracker_json', type=str, default='temp.json')

    args = parser.parse_args()

    image = cv2.imread(args.image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    tracker = Template_matching_based_tracker()

    # in ferns, uv corner is defined as:
    # detector->u_corner[0] = 0;                                detector->v_corner[0] = 0;
    # detector->u_corner[1] = detector->model_image->width - 1; detector->v_corner[1] = 0;
    # detector->u_corner[2] = detector->model_image->width - 1; detector->v_corner[2] = detector->model_image->height - 1;
    # detector->u_corner[3] = 0;                                detector->v_corner[3] = detector->model_image->height - 1;

    h, w = gray.shape
    u_corner = [0, w - 1, w - 1, 0]
    v_corner = [0, 0, h - 1, h - 1]
    tracker.learn(gray, 5, 40, 20, 20,
                u_corner[0], v_corner[1], u_corner[2], v_corner[2],
                40, 40, 10000)
    print(tracker)

    tracker.save(args.tracker_file, args.tracker_json)