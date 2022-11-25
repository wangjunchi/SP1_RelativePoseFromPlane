import numpy
import pandas
import os


def main():
    csv_path = 'hypersim_split.csv'
    df = pandas.read_csv(csv_path)
    # filter all train scenes
    train_df = df[df['split_partition_name'] == 'train']
    test_df = df[df['split_partition_name'] == 'test']

    # count the unique scenes
    train_scenes = train_df['scene_name'].unique()
    print("number of train scenes: ", len(train_scenes))
    test_scenes = test_df['scene_name'].unique()
    print("number of test scenes: ", len(test_scenes))

    # sample 20 scenes for test
    test_scenes = numpy.random.choice(test_scenes, 20, replace=False)
    # sort the scenes
    test_scenes = sorted(test_scenes)
    print("number of test scenes: ", len(test_scenes))
    print("test scenes: ", test_scenes)


if __name__ == '__main__':
    main()
