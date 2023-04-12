import os
import shutil

global_dir = "/data/chase/Chase/Lev1/2022/"
target_dir = "temp/"
cnt = 0


def list_dir(dir_path: str):
    arr = os.listdir(dir_path)
    global cnt
    for path in arr:
        path = dir_path + '/' + path
        if os.path.isdir(path):
            list_dir(path)
        if os.path.isfile(path):
            filename = path.split('/')[-1]
            if filename.split('.')[-1] == 'png' or filename.split('.')[-1] == 'PNG':
                if filename.split('_')[-1].split('.')[0] == 'HA':
                    shutil.copy(path, target_dir + '/' + filename)
                    cnt = cnt + 1
            else:
                pass


if __name__ == '__main__':
    list_dir(global_dir)
    print("done! with" + str(cnt) + " HA png file!")
