import sys

print("\nHello,\nChoose one of the following procedures: \n")

def input_console(string, n_options):
    while True:

        information = input(string)
        if information.isdigit():
            if int(information) in range(1, n_options+1):
                break
        else:
            print('\nInput out of scope of possibilites.\n')
    
    return int(information)-1


MODE = input_console("[1] - Feature Extraction \n[2] - Training \n[3] - Evaluation \n[4] - Inference\n\n>", 4)

if MODE == 0:

    sys.path.insert(1, './src/features/')
    import build_features

    dataset_i = input_console("\nChoose a dataset:\n\n[1] - DocLayNet\n\n>", 1)
    part_i = input_console("\nChoose a partition of the dataset to extract:\n\n[1] - Train\n[2] - Test\n[3] - Validation\n[4] - All\n\n>", 4)
    mode_i = input_console("\nChoose the type of model you intend to work with:\n\n[1] - Vision\n[2] - Multimodal\n\n>", 2)
    noise_i = input_console("\nChoose how would you like to organize the labels:\n\n[1] - Default\n[2] - Binary\n[3] - Triplet\n\n>", 3)

    dataset = ['doclaynet']
    part = ['train', 'test', 'val', 'all']
    mode = ['vision', 'multimodal']
    noise = ['default', 'binary', 'triplet']

    DATASET, PART, MODE, NOISE, SAVE_TYPE = dataset[dataset_i], part[part_i], mode[mode_i], noise[noise_i], 'json'

    if PART != 'all':
        build_features.run(DATASET, MODE, PART, SAVE_TYPE, NOISE)
    else:
        build_features.run(DATASET, MODE, part[0], SAVE_TYPE, NOISE)
        build_features.run(DATASET, MODE, part[1], SAVE_TYPE, NOISE)
        build_features.run(DATASET, MODE, part[2], SAVE_TYPE, NOISE)

if MODE == 1:

    sys.path.insert(1, './src/models/')
    import build_features


