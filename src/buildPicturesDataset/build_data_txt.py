import os

def make_txt(path):
    all_dirs = [dir for dir in os.listdir(path) if os.path.isdir(os.path.join(path, dir))]
    lables = range(len(all_dirs))
    with open(os.path.join(path, 'data.txt'), 'a', encoding='utf-8') as f:
        for i, dir in enumerate(all_dirs):
            all_files = [file for file in os.listdir(os.path.join(path, dir)) if os.path.isfile(os.path.join(path, dir, file))]
            for file in all_files:
                f.write(file + ' ' + str(i) + ' ' + dir + '\n')
if __name__ == '__main__':
    root = r'F:\code\AI\wx_with_ly_DeepLearning_Record\dataset\flower_photos'
    make_txt(root)