import argparse
import time
import numpy as np
from src.dataset.POT280Data import POT280Data, VideoData
from pathlib import Path


def main(args):
    dataset = POT280Data('POT280Data', 'annotation')

    sigmoid_diff_score = lambda x: 1 / (1 + np.exp(0.5 * x - 4))

    # create unique result file, any better way?
    i = 0
    while 1:
        result_path = Path('results/POT208_test_result_{}.txt'.format(i))
        if result_path.exists():
            i+=1
            continue
        else:
            with open(result_path.as_posix(), 'w') as f:
                f.write('test time: {}'.format(time.strftime("%Y-%m-%d %H:%M:%S\n", time.localtime())))
            break

    failure_case = 0
    success_case = 0
    for i in range(1, 41):
        video_path, homography_path, points_path = dataset.get_video_with_annotation(i)
        video_data = VideoData(video_path, homography_path, points_path)
        if not video_data.valid:
            continue

        if not video_data.init_first_frame():
            failure_case += 1
            continue

        success, count, failure_count, diffs = video_data.track(args.visualize)

        std = 0
        mean = 0
        score_one_video = 0
        if not success:
            failure_case += 1
            # delete the checkpoint file. In this way, the next time we meet this video, it will train again
            if video_data.checkpoint_path.exists():
                video_data.checkpoint_path.unlink()
        else:
            success_case += 1

            if diffs.shape[0] > 1:
                std = diffs.std()
                mean = diffs.mean()

            success_rate_one_video = (count - failure_count) / count
            score_one_video = 0.5 * (success_rate_one_video + 1) * sigmoid_diff_score(diffs).mean()

        with open(result_path.as_posix(), 'a') as f:  # append to file
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(i, count, failure_count, mean, std, score_one_video))

        video_data.dispose()

    # summary
    with open(result_path.as_posix(), 'a') as f:
        f.write('\n\nsummary:\n')
        f.write('failure case: {}/40\n'.format(failure_case))
        f.write('success case rate: {}\n'.format(success_case / (failure_case + success_case)))
        f.write('average diff: {}\n'.format(mean))
        f.write('std of diff: {}\n'.format(std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--visualize', help='visualize tracking result, default is False', action='store_true')
    args = parser.parse_args()

    main(args)