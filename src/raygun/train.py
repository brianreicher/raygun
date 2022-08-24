from glob import glob
from importlib import import_module
import os
from subprocess import call
import sys

from .read_config import read_config
from .utils import get_config_name

def cluster_train(config_path=None):
    """Train system via cluster job (available through CLI as raygun-train-cluster)

    Args:
        config_path (str, optional): Path to json file for training configuration. Defaults to command line argument.
    """

    config = read_config(config_path)
    os.chdir(os.path.dirname(config_path))

    command = config["job_command"] + [f"-o {os.path.dirname(config_path)}/train.out", "raygun-train", config_path, "&"]
    try:
        retcode = call(command, shell=True)
        if retcode < 0:
            print("Child was terminated by signal", -retcode, file=sys.stderr)
        else:
            print("Child returned", retcode, file=sys.stderr)
    except OSError as e:
        print("Execution failed:", e, file=sys.stderr)


def train(config_path=None):
    """Train system (available through CLI as raygun-train)

    Args:
        config_path (str, optional): Path to json file for training configuration. Defaults to command line argument.
    """

    if config_path is None:
        config_path = sys.argv[1]
    
    config = read_config(config_path)
    System = import_module('.'.join(['raygun', config['framework'], 'systems', config['system']]))
    system = System(config_path)
    
    system.logger.info('System loaded. Training...')
    _ = system.train()
    system.logger.info('Done training!')


def _batch_train(folder):
    os.chdir(folder)
    
    subfolders = glob('*/')
    if len(subfolders) > 0:
        config_paths = []
        for subfolder in subfolders:
            config_paths += _batch_train(subfolder)        

    else:
        config_path = os.path.realpath('train_conf.json')
        if os.path.exists(config_path):
            config = read_config(config_path)
            
            if 'job_command' in config.keys():
                cluster_train(config_path)
            else:
                train(config_path)
            
            config_paths = [config_path]
        
        else:
            config_paths = []
            
    os.chdir('..')
    return config_paths


def batch_train(base_folder=None):
    """Batch train systems (available through CLI as raygun-train-batch).

    Args:
        base_folder (str, optional): Path to folder containing base training configuration json file and nested folders for batch training. Defaults to command line argument.
    """    
    
    if base_folder is None:
        base_folder = sys.argv[1]        
    
    os.chdir(base_folder)
    base_folder = os.getcwd() # get absolute path

    config_paths = _batch_train('.')

    for config_path in config_paths:
        config_name = get_config_name(config_path, base_folder)

        if not os.path.islink(os.path.join(base_folder, 'tensorboards',config_name)):
            os.symlink(os.path.join(os.path.dirname(config_path), 'tensorboard'), 
                    os.path.join(base_folder, 'tensorboards', config_name),
                    target_is_directory=True)


if __name__ == '__main__':
    train(sys.argv[1])