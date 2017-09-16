
import helper.helper as helper
from functions import normalize, one_hot_encode

cifar10_dataset_folder_path = 'cifar-10-batches-py'

# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)