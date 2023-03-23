import sys

def discrete_console(text, n_options):

    while True:
        information = input(text)
        if information.isdigit():
            if int(information) in range(1, n_options+1):
                break
            else:
                print('\nInput out of scope of possibilites. Try again.\n')
        else:
            print('\nInput is not a digit. Try again.\n')
    
    return int(information)-1

def open_console(text, raw_type):

    while True:
        information = input(text)
        if raw_type == int:
            if information.isdigit():
                return int(information)
            else:
                print('\nInput is not a digit. Try again.\n')

        if raw_type == str:
            if not information.isdigit():
                return information
            else:
                print('\nInput is not a string. Try again.\n')
    


print("\nHello,\nChoose one of the following procedures: \n")

MODE = discrete_console("[1] - Feature Extraction \n[2] - Training \n[3] - Evaluation \n[4] - Inference\n\n>", 4)

if MODE == 0:

    sys.path.insert(1, './src/features/')
    import build_features

    datasets = ['doclaynet']
    parts = ['train', 'test', 'val', 'all']
    modes = ['vision', 'multimodal']
    noises = ['default', 'binary', 'triplet']

    dataset_i = discrete_console("\nChoose a dataset:\n\n[1] - DocLayNet\n\n>", 1)
    part_i = discrete_console("\nChoose a partition of the dataset to extract:\n\n[1] - Train\n[2] - Test\n[3] - Validation\n[4] - All\n\n>", 4)
    mode_i = discrete_console("\nChoose the type of model you intend to work with:\n\n[1] - Vision\n[2] - Multimodal\n\n>", 2)
    noise_i = discrete_console("\nChoose how would you like to organize the labels:\n\n[1] - Default\n[2] - Binary\n[3] - Triplet\n\n>", 3)

    dataset, part, mode, noise, save_type = datasets[dataset_i], parts[part_i], modes[mode_i], noises[noise_i], 'json'

    if part != 'all':
        build_features.run(dataset, mode, part, save_type, noise)
    else:
        build_features.run(dataset, mode, parts[0], save_type, noise)
        build_features.run(dataset, mode, parts[1], save_type, noise)
        build_features.run(dataset, mode, parts[2], save_type, noise)

if MODE == 1:
    
    sys.path.insert(1, './src/models/')

    noises = ['default', 'binary', 'triplet']

    model_i = discrete_console("\nChoose the model you intend to train:\n\n[1] - Mask R-CNN\n[2] - LayoutLM\n[3] - LayoutLMv3\n\n>", 3)
    noise_i = discrete_console("\nChoose the noise labelling you intend to use:\n\n[1] - Default\n[2] - Binary\n[3] - Triplet\n\n>", 3)
    n_epochs = open_console("\nChoose the number of epochs that will be performed:\n\n>", int)
    repository_id = open_console("\nChoose a name for your model (it will be saved in the /models folder):\n\n>", str)

    noise = noises[noise_i]

    if model_i == 0: #MaskRCNN TODO
        None

    if model_i == 2: #LayoutLMV3 TODO
        None

    if model_i == 1: #LayoutLMv1

        import train_layoutlm

        train_part = open_console("\nChoose the percentage of training data that will be used for training (a number from 1 to 100):\n\n>", int)
        hf_hub = discrete_console("\nWould you like to connect to the HuggingFace hub?\n\n[1] - No\n[2] - Yes\n\n>", 2)

        if hf_hub == 1:
            hf_hub_id = open_console("\nPlease select a name for your model at HuggingFace's hub:\n\n>", str)
            hf_hub_token = open_console("\nPlease insert your HuggingFace token:\n\n>", str)
        else:
            hf_hub_id, hf_hub_token = None, None
        
        train_layoutlm.run(noise, str(train_part), n_epochs, repository_id, hf_hub, hf_hub_id, hf_hub_token)
    




    
        




