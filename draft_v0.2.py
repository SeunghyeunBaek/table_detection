from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = 'img/'
proc_path = 'proc_img/'
filename = 'image02.png'
sp_filename = filename.split('.')
proc_filename = sp_filename[0] + '_proc.' + sp_filename[1]


def get_ax_mean(axis, img):
	ax_mean_ls = list()
	img_dim = img.shape
	ax = 0 if axis == 'row' else 1
	idx_ls = [i for i in range(img_dim[ax])]
	for i in idx_ls:
		line = img[i, :] if ax == 0 else img[:, i]
		line_mean = np.mean(line)
		ax_mean_ls.append(line_mean)
	return dict(zip(idx_ls, ax_mean_ls))

def dict_plot(dict_, img, axis):
	aspect = img.shape[0]/img.shape[1]
	fig, ax = plt.subplots(2, figsize=(12, 9))
	items = sorted(dict_.items())
	x, y = zip(*items)
	if axis == 'row':
		ax[0].plot(y, x)
		ax[0].invert_yaxis()
	else: ax[0].plot(x, y)
	ax[1].imshow(img, cmap='gray')
	plt.show()


img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)
img_dim = img.shape

# 이진화
bi_thr = 200
ret, bi_img = cv2.threshold(img, bi_thr, 255, cv2.THRESH_BINARY) 
cv2.imwrite(proc_path+proc_filename, bi_img)

# 픽셀값 평균
row_mean_dict = get_ax_mean(axis='row', img=bi_img)  # row 픽셀값 평균
col_mean_dict = get_ax_mean(axis='col', img=bi_img)  # col 픽셀값 평균

# 텍스트 영역 지정
# dict_plot(row_mean_dict, bi_img, 'row')
dict_plot(row_mean_dict, bi_img, 'row')

# plt.plot(x=row_mean_dict.keys(), y=row_mean_dict.values())






print(img_dim)


# cv2.imshow('img', bi_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()