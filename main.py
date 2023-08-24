# self-developed
import data_preprocessing

# python-library
import shutil
import os
import json
import pandas as pd
import numpy as np
import cv2

if __name__ == '__main__':
    # 데이터셋 경로 설정
    dataset_path = '생활 폐기물 이미지'

    # 데이터 저장 경로 설정
    data_path = 'data'

    # 클래스 정보 딕셔너리
    classes = {'15': '비닐', '17': '유리병', '21': '종이류', '22': '캔류', '23': '페트병', '24': '플라스틱류'}
    classes_translation = {'비닐': 'vinyl', '유리병': 'glass', '종이류': 'paper', '캔류': 'can', '페트병': 'PET', '플라스틱류': 'plastic'}
    classes_to_code = {'vinyl': '15', 'glass': '17', 'paper': '21', 'can': '22', 'PET': '23', 'plastic': '24'}

    # 이미지 속성 딕셔너리
    translation = {
        '주간': 'day',
        '야간': 'night',
        '실내': 'indoor',
        '실외': 'outdoor',
        '스튜디오': 'studio',
        '비닐류': 'vinyl',
        '유리병류': 'glass',
        '종이류': 'paper',
        '캔류': 'can',
        '페트병류': 'PET',
        '플라스틱류': 'plastic'
    }

    # 확장자
    image_ext = ['jpg', 'png']
    json_ext = ['json', 'Json']

    # 라벨링 데이터 내에 필요없는 클래스의 자료들 삭제
    data_preprocessing.folders_by_classes(f'{dataset_path}/Training/Training_라벨링데이터')
    data_preprocessing.folders_by_classes(f'{dataset_path}/Validation/[V라벨링]라벨링데이터')

    # 파일 이동
    data_preprocessing.gather_data(dataset_path)

    # 누락 데이터 제거
    # 1차 json 파일 리스트
    json_files = [x for x in os.listdir(data_path) if x.split('.')[-1] in json_ext]

    # 필요 이미지 속성 리스트
    headers = ['MAKE', 'Camera Model Name', 'resolution', 'iso', 'daynight', 'place', 'obj_class', 'obj_color', 'bbox']

    # 이미지 속성 값 추출
    image_properties = {}
    removed_files = {}
    for f in json_files:
        values = []
        values_lst = []
        errors = []
        # json 파일 불러오기
        with open(os.path.join(data_path, f), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        # 객체 1개가 아닐경우, 파일 삭제
        if data['BoundingCount'] != '1':
            errors.append('multiple object detected')
            removed_files[f] = errors
            os.remove(os.path.join(data_path, f))
            continue   
        # 객체 1개일 경우, 필요 이미지 속성 값 추출하여 저장
        else:
            if data['Bounding'][0]['Drawing'] == 'BOX':
                values.append(data['MAKE'])
                values.append(data['Camera Model Name'])
                values.append(data_preprocessing.resolution_preprocess(data['RESOLUTION']))
                values.append(int(data['Sensitivity iso']))
                values.append(translation[data['DAY/NIGHT']])
                values.append(translation[data['PLACE']])
                values.append(translation[data['Bounding'][0]['CLASS']])
                obj_color = list(map(int,data['Bounding'][0]['Color'].split('/')))
                values.append(data_preprocessing.rgb_to_hex(*obj_color))
                values_lst.append(data['Bounding'][0]['x1'])
                values_lst.append(data['Bounding'][0]['y1'])
                values_lst.append(data['Bounding'][0]['x2'])
                values_lst.append(data['Bounding'][0]['y2'])
                values.append(values_lst)
            else:
                errors.append('polygon label detected')
                removed_files[f] = errors
                os.remove(os.path.join(data_path, f))
                continue
        image_properties[f] = values

    # 이미지 속성값 csv파일로 저장
    df = pd.data_dframe.from_dict(image_properties, orient='index', columns=headers)
    df.to_csv(os.path.join('data1.csv'), encoding='cp949')

    # 제거할 이미지 csv파일로 저장
    df2 = pd.data_dframe.from_dict(removed_files, orient='index', columns=['reason'])
    df2.to_csv(os.path.join('removed_data1.csv'), encoding='cp949')

    # 파일 리스트
    files = [x for x in os.listdir(data_path) if x.split('.')[-1] in (image_ext + json_ext)]
    image_files = [x.split('.')[0] for x in files if x.split('.')[-1] in image_ext]
    json_files = [x.split('.')[0] for x in files if x.split('.')[-1] in json_ext]

    # 대칭 차집합 // json 파일만 있거나 image 파일만 있는 데이터 리스트
    unused = set(json_files) ^ set(image_files)

    # 필요없는 데이터 제거
    for f in files:
        ftype = f.split('.')[-1]
        if f.split('.')[0] in unused:
            os.remove(os.path.join(data_path, f))
        # else:
        #     if ftype.lower() in image_ext:
        #         shutil.move(os.path.join(data_path, f), os.path.join(data_path, 'image', f))
        #     elif ftype.lower() in json_ext:
        #         shutil.move(os.path.join(data_path, f), os.path.join(data_path, 'label', f))

    # 디렉터리가 없을 경우, 생성
    if not os.path.isdir(os.path.join(data_path, 'train')):
        data_preprocessing.create_folders(data_path)

    # csv파일 정보를 기반으로 텍스트 파일 생성 // img_size, bbox,  obj_class
    data_df = pd.read_csv('data1.csv')
    label_data_path = os.path.join(data_path, 'jsons')
    j_fs = [x for x in os.listdir(label_data_path) if x.split('.')[-1] in json_ext]
    for j_f in j_fs:
        row = data_df[data_df.iloc[:,0] == j_f]
        img_size = list(map(int, row['resolution'].values[0].split('*')))
        x1, y1, x2, y2 = list(map(int, ast.literal_eval(row['bbox'].values[0])))
        bbox = [x1, x2, y1, y2]
        obj_class = row['obj_class'].values[0]
        x,y,w,h = data_preprocessing.bboxtoyoloformat(img_size, bbox)
        with open(f'{os.path.join(data_path, j_f.split(".")[0])}.txt', 'w') as txtfile:
            txtfile.write(f'{classes_to_code[obj_class]} {x} {y} {w} {h}')

    # 파일을 8:1:1 비율로 분배
    json_files = [x for x in os.listdir(data_path) if x.split('.')[-1] in ['Json', 'json']]
    image_files = [x for x in os.listdir(data_path) if x.split('.')[-1] in ['jpg', 'png']]

    train_files, val_files, test_files = np.split(np.array(image_files), [int(len(image_files)*0.8), int(len(image_files)*0.9)])
    all_files = {'train': train_files, 'val': val_files, 'test': test_files}
    for key, value in all_files.items():
        data_preprocessing.move_by_dirs(key, data_path, value)

    for j_f in json_files:
        shutil.move(os.path.join(data_path, j_f), os.path.join(data_path, 'jsons', j_f))

    #image/label folder의 데이터를 data_path로 이동시키기
    sub_dir = 'train'
    image_data_path = os.path.join(data_path, sub_dir, 'images')
    label_data_path = os.path.join(data_path, sub_dir, 'labels')

    images = [x for x in os.listdir(image_data_path)]
    labels = [x for x in os.listdir(label_data_path)]

    for image in images:
        shutil.move(os.path.join(image_data_path, image), os.path.join(data_path, image))

    for label in labels:
        shutil.move(os.path.join(label_data_path, label), os.path.join(data_path, label))

    # 이미지 데이터명 변경 
    sub_dir = ['train', 'test', 'val']
    for sub in sub_dir:
        image_data_path = os.path.join(data_path, sub, 'images')
        image_datas = [x for x in os.listdir(image_data_path)]
        for image_data in image_datas:
            os.rename(os.path.join(image_data_path, image_data), os.path.join(image_data_path, image_data.split('.')[0] + '_copy.jpg'))

    # 이미지 데이터 resize 및 원본데이터 imgs 폴더로 이동
    sub_dir = ['train', 'test', 'val']
    for sub in sub_dir:
        image_data_path = os.path.join(data_path, sub, 'images')
        image_datas = [x for x in os.listdir(image_data_path) if x.endswith('_copy.jpg')]
        for image_data in image_datas:
            img = cv2.imread(os.path.join(image_data_path, image_data))
            resized_img = cv2.resize(img, dsize=(640,640), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(image_data_path, '_'.join(image_data.split('_')[:-1]) + '.jpg'), resized_img)
            shutil.move(os.path.join(image_data_path, image_data), os.path.join(data_path, 'imgs', f"{'_'.join(image_data.split('_')[:-1])}.jpg"))

    # 작업 종료시 슬랙에 메시지 전송
    from slackbot_run import SlackAPI
    token = 'xoxb-5713411137014-5773768770325-TvV4SRyB8WMOwl7yh3SO69Dx'
    slack = SlackAPI(token)
    channel_name = 'slackbot_'
    text = '끝'
    channel_id = slack.get_channel_id(channel_name)
    slack.post_message(channel_id, text)