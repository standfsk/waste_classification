import os
import shutil
import json
import numpy as np
import pandas as pd
from tqdm import tqdm

# 값이 디렉터리 인지 확인
def check_dir(path):
    return os.path.isdir(path)

# 디렉터리 리스트 
def list_dir(path):
    return [x for x in os.listdir(os.path.join(path)) if check_dir(os.path.join(path, x))]

# 파일 이동
def move_files(src, dest, exts):
    files = [x for x in os.listdir(src) if x.split('.')[-1] in (exts['image_ext'] + exts['json_ext'])]
    for f in files:
        shutil.move(os.path.join(src, f), os.path.join(dest, f))

# 파일 복사
def copy_files(src, dest, exts):
    files = [x for x in os.listdir(src) if x.split('.')[-1] in (exts['image_ext'] + exts['json_ext'])]
    for f in tqdm(files, desc='파일 복사'):
        shutil.copy(os.path.join(src, f), os.path.join(dest, f))

# 클래스 정보에 해당하지 않는 폴더들 삭제
def folders_by_classes(path, classes):
    for dir in [x for x in os.listdir(path) if x not in classes.values()]:
        shutil.rmtree(os.path.join(path, dir))

# 디렉터리별 파일을 /data로 이동
def gather_data(path, data_path, exts):
    if list_dir(path):
        for path2 in list_dir(path):
            gather_data(os.path.join(path, path2))
    else:
        move_files(path, data_path, exts)

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

# 디렉터리별로 데이터 분배
def move_by_dirs(path, dir, files):
    img_path = os.path.join(path, dir, 'images')
    lbl_path = os.path.join(path, dir, 'labels')
    for f in files:
        txtfile = f.split('.')[0] + '.txt'
        shutil.move(os.path.join(path, f), os.path.join(img_path, f))
        shutil.move(os.path.join(path, txtfile), os.path.join(lbl_path, txtfile))

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
    bbox_resized = list(map(round, [x1*x_ratio, x2*x_ratio, y1*y_ratio, y2*y_ratio]))
    return bbox_resized

# 이미지 데이터 표준화
def normalize_image(image):
    normalized_r = image[:, :, 0] / 255.0
    normalized_g = image[:, :, 1] / 255.0
    normalized_b = image[:, :, 2] / 255.0

    normalized_image = np.stack((normalized_r, normalized_g, normalized_b), axis=-1)
    return normalized_image

# 이미지 속성 값 추출 -> csv
def extract_imgattr(path, translation, headers):
    files = set([x.split('.')[0] for x in os.listdir(path) if x.endswith('.Json')])

    image_properties = {}
    removed_files = {}
    for f in tqdm(files, desc='속성 추출'):
        values = []
        errors = []
        # json 파일 불러오기
        with open(os.path.join(path, f+'.Json'), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        # 객체 1개가 아닐경우, 파일 삭제
        if data['BoundingCount'] != '1':
            errors.append('multiple object detected')
            removed_files[f] = errors
            os.remove(os.path.join(path, f+'.Json'))
            os.remove(os.path.join(path, f+'.jpg'))
            continue
        # 객체 1개일 경우, 필요 이미지 속성 값 추출하여 저장
        else:
            if data['Bounding'][0]['Drawing'] == 'BOX':
                # values.append(data['MAKE'])
                # values.append(data['Camera Model Name'])
                values.append(resolution_preprocess(data['RESOLUTION']))
                values.append(int(data['Sensitivity iso']))
                values.append(translation[data['DAY/NIGHT']])
                values.append(translation[data['PLACE']])
                values.append(translation[data['Bounding'][0]['CLASS']])
                obj_color = list(map(int, data['Bounding'][0]['Color'].split('/')))
                values.append(rgb_to_hex(*obj_color))
                values.append(data['Bounding'][0]['x1'])
                values.append(data['Bounding'][0]['y1'])
                values.append(data['Bounding'][0]['x2'])
                values.append(data['Bounding'][0]['y2'])
            else:
                errors.append('polygon label detected')
                removed_files[f] = errors
                os.remove(os.path.join(path, f+'.Json'))
                os.remove(os.path.join(path, f+'.jpg'))
                continue
        image_properties[f] = values

    # 이미지별 속성값 csv파일로 저장
    df = pd.DataFrame.from_dict(image_properties, orient='index', columns=headers)
    df.to_csv(os.path.join(path, 'data1.csv'), encoding='cp949')

    # 제거 이미지 및 제거 이유 csv파일로 저장
    df2 = pd.DataFrame.from_dict(removed_files, orient='index', columns=['reason'])
    df2.to_csv(os.path.join(path, 'removed_data1.csv'), encoding='cp949')

# 필요없는 데이터 제거
def remove_unmatched(path, exts, error_imgs):
    from collections import Counter
    files = Counter([x.split('.')[0] for x in os.listdir(path) if x.split('.')[-1] in (exts['image_ext'] + exts['json_ext'])])
    unmatched_files = [key for key, value in files.items() if value == 1]
    
    for f in unmatched_files:
        try:
            os.remove(os.path.join(path, f+'.jpg'))
        except:
            os.remove(os.path.join(path, f+'.Json'))
    
    for f in error_imgs:
        if os.path.exists(path, f+'.jpg'):
            os.remove(os.path.join(path, f+'.jpg'))
            os.remove(os.path.join(path, f+'.Json'))

# bbox value를 YOLO format으로 변환
def cvt2YOLO(img_size, bbox):
    dw = 1./img_size[0]
    dh = 1./img_size[1]
    x = (bbox[0] + bbox[1])/2.0
    y = (bbox[2] + bbox[3])/2.0
    w = bbox[1] - bbox[0]
    h = bbox[3] - bbox[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x,y,w,h)
    
# 데이터 전처리
def data_preprocess(path, classes_to_code, img_size, exts):
    from PIL import Image
    import cv2

    # csv파일 정보를 기반으로 텍스트 파일 생성 // img_size, bbox,  obj_class
    files = set([x.split('.')[0] for x in os.listdir(path) if x.split('.')[-1] in (exts['image_ext'] + exts['json_ext'])])
    for f in tqdm(files, desc='데이터 전처리'):
        image_file = f + '.jpg'
        # 이미지 데이터 resize
        img = Image.open(os.path.join(path, image_file))
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        resized_img = cv2.resize(img, dsize=(img_size), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(path, image_file), resized_img)
        # 라벨 데이터 전처리
        bbox_resized = resize_bbox(img, resized_img, os.path.join(path, image_file.split('.')[0]+'.json'))
        # change_bbox(bbox_resized, os.path.join(path, image_file.split('.')[0]+'.json'), os.path.join(path, image_file.split('.')[0]+'.json'))
        # json -> txt
        YOLO_values = cvt2YOLO(img_size, bbox_resized)
        with open(f'{os.path.join(path, f)}.txt', 'w') as txtfile:
            txtfile.write(f'{f.split("_")[0]} {YOLO_values[0]} {YOLO_values[1]} {YOLO_values[2]} {YOLO_values[3]}')
        # 라벨 데이터 제거
        os.remove(os.path.join(path, f+'.Json'))
        

# 데이터 분배 (8:1:1)
def distribute_files(path, exts):
    from sklearn.model_selection import train_test_split
    images = [x for x in os.listdir(path) if x.split('.')[-1] in exts['image_ext']]
    labels = [x.split('_')[0] for x in images]
    train_input, test_input, train_target, test_target = train_test_split(images, labels, test_size=0.1, random_state=42, shuffle=True, stratify=labels)
    train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.1, random_state=42, shuffle=True, stratify=train_target)
    all_files = {'train': train_input, 'val': val_input, 'test': test_input}
    for dir, file_list in tqdm(all_files.items(), desc='데이터 분배'):
        move_by_dirs(path, dir, file_list)

# 슬랙 봇에 메시지 보내기
def end_message(token, text):
    from slackbot_run import SlackAPI
    slack = SlackAPI(token)
    channel_name = 'slackbot_'
    channel_id = slack.get_channel_id(channel_name)
    slack.post_message(channel_id, text)