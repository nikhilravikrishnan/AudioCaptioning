# Config File Format

List of parameters:
- system_settings
    - **data_dir** *(string)* - The path to the directory where the Clotho datatset is stored
    - **save_dir** *(string)* - The path to the directory to save model checkpoints and metrics
    - **sys_path** *(string)* - The path to the directory containing all the project code
    - **model_lib_path** *(string)* - The path to the directory containing the Python modules for our models
    - **checkpoint_path** *(string)* - Where the checkpoint of a trained model to evaluate has been stored

- hyperparameters
    - **epochs** *(int)* - The number of epochs to use for training the model
    - **batch_size** *(int)* - How many items to include in each batch when training the model
    - **eval_batch_size** *(int)* - How many items to include in each batch when evaluating the model
    - **lr** *(float)* - The starting learning rate for the model

- network
    - **model** *(string)* - The name of the model type to use for prediction or training (ResNet, ViT, or PANN)
    - **fine_tune** *(boolean)* - Whether to train additional layers in the model other than the alignment heads

- weights
    - **pretrained** *(boolean)* - Whether or not to use a pretrained model
    - **retrieval_method** *(string)* - Whether the model's weights should be retrieved `from_library` (i.e. PyTorch or HuggingFace) or `from_file`
    - **weight_file** *(string)* - The file path for a model's weights if they needed to be loaded from a file
