import glob
import os

dirname = os.path.dirname(__file__)
urfd_path = os.path.join(dirname, '../datasets/URFD_opticalflow')
sisfall_path = os.path.join(dirname, '../datasets/SisFall_OF')
L = 10

content = [
    ['URFD', urfd_path],
]

# URFD and Sisfall
for dataset_name, dataset_path in content[:2]:
    falls = len(glob.glob(dataset_path + 'Falls/*'))
    notfalls = len(glob.glob(dataset_path + 'NotFalls/*'))
    print('-'*10)
    print(dataset_name)
    print('-'*10)
    print('Fall sequences: {}, ADL sequences: {} (Total: {})'.format(
        falls, notfalls, falls + notfalls)
    )

    falls = glob.glob(dataset_path + 'Falls/*')
    fall_stacks = sum(
        [len(glob.glob(fall +'/*')) - L + 1 for fall in falls]
    )
    fall_frames = sum(
        [len(glob.glob(fall +'/*')) for fall in falls]
    )
    notfalls = glob.glob(dataset_path + 'NotFalls/*')
    nofall_stacks = sum(
        [len(glob.glob(notfall +'/*')) - L + 1 for notfall in notfalls])
    nofall_frames = sum(
        [len(glob.glob(notfall +'/*'))  for notfall in notfalls])

    print('Fall stacks: {}, ADL stacks: {} (Total: {})'.format(
        fall_stacks, nofall_stacks, fall_stacks + nofall_stacks)
    )
    print('Fall frames: {}, ADL frames: {} (Total: {})\n\n'.format(
        fall_frames, nofall_frames,fall_frames + nofall_frames)
    )
