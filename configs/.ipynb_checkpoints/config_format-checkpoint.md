# Config File Format

List of parameters:
- system_settings
    - **data_dir** *(string)* - The path to the directory where the Clotho datatset is stored
    - **save_dir** *(string)* - The path to the directory to save model checkpoints and Tensorboard statistics
    - **sys_path** *(string)* - The path to the directory containing all the project code
    - **model_lib_path** *(string)* - The path to the directory containing the Python modules for our models
- hyperparameters
    - **split** *(string)* - The split (dev or test) to use for model training or testing 
    - **epochs** *(int)* - The number of epochs to use for training the model
    - **batch_size** *(int)* - How many items to include in each batch when training and evaluating the model
- network
    - **model** *(string)* - The name of the model type to use for prediction or training
    - **get_feat_vec** *(boolean)* - Whether to get the feature vector by retrieving the weights at an intermediate layer of the network
    - **embedding_layer** *(string)* - The Module property that contains the intermediate layer of the network from which to retrieve a feature vector for an image
- weights
    - **pretrained** *(boolean)* - Whether or not to use a pretrained model
    - **retrieval_method** *(string)* - Whether the model's weights should be retrieved `from_library` (i.e. PyTorch or HuggingFace) or `from_file`
    - **weight_file** *(string)* - The file path for a model's weights if they needed to be loaded from a file
