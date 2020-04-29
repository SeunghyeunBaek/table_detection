from PIL import Image
import pytesseract as pt
import csv
import cv2
import pandas as pd

pt.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


path = 'img/'
filename = 'image02.png'
image = Image.open(path+filename)
save_path = 'res/'
filename_split = filename.split('.')
save_filename = filename_split[0] + '_res.' + filename_split[1]
save_res = filename_split[0] + '.csv'

custom_oem_psm_config = r'--oem 1 --psm 4'
res = pt.image_to_data(image,
						 lang='kor',
						 config=custom_oem_psm_config,
						 output_type=pt.Output.DICT)

n_box = len(res['level'])  # 인식 글자 수
img = cv2.imread(path+filename)

for i in range(n_box):
	if res['text'][i] != '':
		(x, y, w, h) = (res['left'][i], res['top'][i], res['width'][i], res['height'][i])
		cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)	



# save
print(f'save..{save_path+save_filename}')
cv2.imwrite(filename=save_path+save_filename, img=img)
res_df = pd.DataFrame(res)
res_df.to_csv(save_path+save_res, encoding='cp949')

# # txt = pt.image_to_data(image,
# 						 lang='kor',
# 						 config=custom_oem_psm_config)

# print(txt)


# txt = pt.image_to_data(image,
# 						 lang='kor',
# 						 config=custom_oem_psm_config)

# import pytesseract
# from pytesseract import Output
# import cv2

# img = cv2.imread(path+filename)

# d = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_oem_psm_config)

# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# cv2.imshow('img', img)
# cv2.waitKey(0)
