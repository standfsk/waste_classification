import os
import shutil
import json
import cv2
import numpy as np

# 값이 디렉터리 인지 확인
def check_dir(path):
    return os.path.isdir(path)

# 디렉터리 리스트 
def list_dir(path):
    return [x for x in os.listdir(os.path.join(path)) if check_dir(os.path.join(path, x))]

# 파일 이동
def move_files(src, dest):
    files = [x for x in os.listdir(src) if not os.path.isdir(os.path.join(src, x))]
    for f in files:
        shutil.move(os.path.join(src, f), os.path.join(dest, f))

# 클래스 정보에 해당하지 않는 폴더들 삭제
def folders_by_classes(path, classes):
    for dir in [x for x in os.listdir(path) if x not in classes.values()]:
        shutil.rmtree(os.path.join(path, dir))

# 디렉터리별 파일을 /data로 이동
def gather_data(path, data_path):
    if list_dir(path):
        for path2 in list_dir(path):
            gather_data(os.path.join(path, path2))
    else:
        move_files(path, data_path)

# r/g/b -> hexadecimal
def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

# resolution 값을 *(asterisk)를 기준으로 큰 값을 *(asterisk) 앞으로 이동
def resolution_preprocess(value):
    value = value.split('*')
    return '*'.join(sorted(list(map(lambda x : round_int(x), value)), reverse=True))

# resolution 값 반올림 (1921 -> 1920, 1929 -> 1930)
def round_int(number):
    return str(round(int(number)/10) * 10)

# train, val, test 디렉터리 생성
def create_folders(path):
    train_path = os.path.join(path, 'train')
    val_path = os.path.join(path, 'val')
    test_path = os.path.join(path, 'test')
    os.makedirs(os.path.join(train_path, 'images'))
    os.makedirs(os.path.join(train_path, 'labels'))
    os.makedirs(os.path.join(val_path, 'images'))
    os.makedirs(os.path.join(val_path, 'labels'))
    os.makedirs(os.path.join(test_path, 'images'))
    os.makedirs(os.path.join(test_path, 'labels'))
    os.mkdir(os.path.join(path, 'jsons'))
    os.mkdir(os.path.join(path, 'imgs'))

# 디렉터리별로 데이터 분배
def move_by_dirs(dir, path, files):
    img_path = os.path.join(path, dir, 'images')
    lbl_path = os.path.join(path, dir, 'labels')
    for f in files:
        txtfile = f.split('.')[0] + '.txt'
        shutil.move(os.path.join(path, f), os.path.join(img_path, f))
        shutil.move(os.path.join(path, txtfile), os.path.join(lbl_path, txtfile))

# bbox 정보를 yolo 형태로 변환
def bboxtoyoloformat(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# bbox의 정보를 변경
def change_bbox(values, json_file, changed_json):
    import json
    x1,y1,x2,y2 = list(map(str, values))
    with open(json_file, 'r', encoding='utf-8') as j_f:
        data = json.load(j_f)
    data['Bounding'][0]['x1'] = x1
    data['Bounding'][0]['y1'] = y1
    data['Bounding'][0]['x2'] = x2
    data['Bounding'][0]['y2'] = y2

    with open(changed_json, 'w', encoding='utf-8') as c_j:
        json.dump(data, c_j, indent=4, ensure_ascii=False)

# 이미지의 resize 비율만큼 bbox값 계산
def resize_bbox(image, resized_image, json_file):
    with open(json_file, 'r', encoding='utf-8') as j_f:
        data = json.load(j_f)
    x1 = int(data['Bounding'][0]['x1'])
    y1 = int(data['Bounding'][0]['y1'])
    x2 = int(data['Bounding'][0]['x2'])
    y2 = int(data['Bounding'][0]['y2'])
    y_ratio = resized_image.shape[0] / image.shape[0]
    x_ratio = resized_image.shape[1] / image.shape[1]
    bbox_resized = list(map(round, [x1*x_ratio, y1*y_ratio, x2*x_ratio, y2*y_ratio]))
    return bbox_resized

# 이미지 데이터 표준화
def normalize_image(image):
    normalized_r = image[:, :, 0] / 255.0
    normalized_g = image[:, :, 1] / 255.0
    normalized_b = image[:, :, 2] / 255.0

    normalized_image = np.stack((normalized_r, normalized_g, normalized_b), axis=-1)
    return normalized_image