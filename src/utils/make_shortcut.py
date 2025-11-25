from pathlib import Path
import os
import errno
def make_link(folder_path):
    root = os.getcwd()
    # find the lastest model config here
    latest_file = max(Path(folder_path).glob('*.h5'), key=os.path.getctime)
    shortcut_traget = str(latest_file)
    shortcut_traget = os.path.join(root, shortcut_traget)
    # make shortcut
    shortcut_name = 'lastest_model.h5'
    # make the path of the shortcut
    newest_model_path = os.path.join(root, folder_path, shortcut_name)

    os.symlink(shortcut_traget, newest_model_path)
    


def force_symlink(folder_path):
    try:
        make_link(folder_path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            lastest = os.path.join(folder_path, 'lastest_model.h5')
            os.remove(lastest)
            make_link(folder_path)
    
if __name__ == "__main__":
    folder_path = 'models'
    print(folder_path)
    force_symlink(folder_path)
