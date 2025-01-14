import streamlit as st
import cv2
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'computer_vision/text-recognize/tesseract/tesseract.exe'
# config_tesseract = '--tessdata-dir computer_vision/text-recognize/tesseract/tessdata --psm 6'

def caixa_texto(i, resultado, img, cor = (255, 0, 0), show_txt=True):
    x = resultado['left'][i]
    y = resultado['top'][i]
    w = resultado['width'][i]
    h = resultado['height'][i]

    cv2.rectangle(img, (x, y), (x+w, y+h), cor, 2)

    if show_txt:
        cv2.putText(img, resultado['text'][i], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, cor, 2)

    return x, y, img

def get_img_data(img, lang, min_conf=40):
    img_copia = img.copy()
    
    if lang == 'Português':
        result = pytesseract.image_to_data(
        img_copia,
        lang='por',
        # config=config_tesseract,
        output_type=pytesseract.Output().DICT
        )
    else:
        result = pytesseract.image_to_data(
        img_copia,
        # config=config_tesseract,
        output_type=pytesseract.Output().DICT
        )

    texto=''
    for i in range(len(result['text'])):
        confianca = float(result['conf'][i])
        if confianca > min_conf:
            texto+=f"{result['text'][i]} "
            x, y, img_copia = caixa_texto(i, result, img_copia)

    st.title('TEXTO IDENTIFICADO:')
    st.write(texto)
    st.image(img)
    st.image(img_copia)




if __name__ == "__main__":
    st.set_page_config(
        page_title="Imagem para texto",
        layout="wide",
        page_icon="",
        )
    
    st.title("Imagem para texto")
    st.write("Projeto criado com o intuito de pegar o texto a partir de uma imagem")
    st.divider()

    idioma = st.radio("Escolha o idioma que vai estar na imagem", options=["Português", "Inglês"])

    uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["png", "jpg", "jpeg", "bmp"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        get_img_data(cv2.cvtColor(image,cv2.COLOR_BGR2RGB), idioma)
        

#streamlit run computer_vision\8-streamlit_text.py
