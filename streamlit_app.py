from PIL import Image, ImageDraw
import streamlit as st
import os
import json

def pil_draw_rect(image, points):
    draw = ImageDraw.Draw(image)
    draw.rectangle(points, outline='red', width=3)
    return image

def main():
    st.set_page_config(layout="wide")
    with st.sidebar.expander("menu", expanded=True):
        pass
        
    folder = st.text_input('폴더명을 입력하세요').strip()
    if folder:
        data_path = os.path.join(folder)
        files = set(x.split('.')[0] for x in os.listdir(data_path))
        file_name = st.selectbox('파일을 선택하세요', files)    

        if file_name != '':
            json_path = os.path.join(data_path, file_name+'.Json')
            image_path = os.path.join(data_path, file_name+'.jpg')
            col1, col2 = st.columns([1.2,1])
            with open(json_path, 'r', encoding='utf-8') as j_f:
                json_data = json.load(j_f)
            points = list(map(int,[json_data['Bounding'][0]['x1'], json_data['Bounding'][0]['y1'], json_data['Bounding'][0]['x2'], json_data['Bounding'][0]['y2']]))
            with col1:
                image = Image.open(image_path)
                st.image(image, width=600)
                labeled_image = pil_draw_rect(image, points)
                st.image(labeled_image, width=600)
            with col2:
                st.json(json_data['Bounding'])

if __name__ == '__main__':
    main()