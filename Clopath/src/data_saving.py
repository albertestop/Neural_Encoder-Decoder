import os

def generate_folder(path):
    items_s = os.listdir(path)
    items = [int(num) for num in items_s]
    if len(items) == 0:
        candidate = 1
    else:
        candidate = 1
        while candidate in items:
                candidate += 1
    new_folder_path = path + '/' + str(candidate)
    os.mkdir(new_folder_path)
    return new_folder_path