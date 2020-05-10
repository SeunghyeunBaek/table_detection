from func2 import *
import cv2
import copy
import numpy as np
import os
import errno

path = 'img/'            # 원본이미지 경로
proc_path = 'proc_img/'  # 이진화 후 이미지 경로
cell_path = 'cell_img/'  # 셀구역을 표현한 이미지
res_path = 'res/' 
split_path = 'sp_img/'   # 셀단위로 쪼개진 이미지
filename = 'image02.png' # 파일명

# 폴더생성
file, ext = filename.split('.')
file_path = file + '/'

try:

    if not(os.path.isdir(res_path+file_path+split_path)):
        print(f'{res_path+file_path+split_path} created')
        os.makedirs(os.path.join(res_path+file_path+split_path))
        
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


def main():
    
    img = cv2.imread(path+filename, cv2.IMREAD_GRAYSCALE)         # 이미지 불러오기
    _, bi_img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)   # 이진화
    _, back_img = cv2.threshold(img, 255, 255, cv2.THRESH_BINARY) # 배경이미지불러오기

    # 직선검출
    erod_size, min_line_size = 3, 31                        # erotion 커널크기, 최소직선크기
    line_dict = get_line(bi_img, erod_size, min_line_size)  # 라인이미지 생성
    row_img = line_dict['row_img']                          # row검출이미지
    col_img = line_dict['col_img']                          # col검출이미지
    
    # 직선박스검출
    line_color = (255, 255, 255)  # 라인표기 색
    line_width = -1               # 라인굵기 -1(영역 내 칠하기)
    
    row_back_img = copy.deepcopy(back_img)
    col_back_img = copy.deepcopy(back_img)
    
    row_dict = get_box(row_img, row_back_img, line_color, line_width, min_line_size, 'row')
    col_dict = get_box(col_img, col_back_img, line_color, line_width, min_line_size, 'col')
    line_img = cv2.add(row_dict['img'], col_dict['img'])
    
    line_file_path = res_path+file_path+file+'_line.'+ext
    print(f'{line_file_path} saved' )
    cv2.imwrite(line_file_path, line_img)
    
    # 직선제거
    rm_line_img = copy.deepcopy(bi_img)
    rm_line_img = cv2.add(rm_line_img, line_img)
    
    rm_line_file_path = res_path+file_path+file+'_remove_line.'+ext
    print(f'{rm_line_file_path} saved' )
    cv2.imwrite(rm_line_file_path, rm_line_img)

    # 잡영제거
    rm_noi_img = remove_noise(rm_line_img, min_line_size)

    # 텍스트 영역검출
    erod_size = 3
    text_img = get_text(rm_noi_img, erod_size, 5)
    
    back_img_text = copy.deepcopy(rm_line_img)
    line_color = (0, 0, 0)
    line_width = 5
    text_box = get_box(text_img, back_img_text, line_color, line_width, min_line_size, 'text')
    text_img = text_box['img']
    
    y_line_ls = sorted(row_dict['coord'])  # row 시작 좌표
    x_line_ls = sorted(col_dict['coord'])  # col 시작 좌표
    
    text_size = 30  # 라인간격이 텍스트 사이즈보다 작을경우 제외
    mod_x_line_ls = get_mod_line(x_line_ls, text_size)
    mod_y_line_ls = get_mod_line(y_line_ls, text_size)
    
    cell_coord = get_cell_coord(mod_x_line_ls, mod_y_line_ls)              # 셀 좌표 계산
    cell_pix_coord_ls = cell_coord['pix_coord']                              # 셀 픽셀 좌표((좌상단, 우하단))
    cell_mat_coord_ls = cell_coord['mat_coord']                              # 셀 행열 좌표(행, 열)
    
    ###
    cell_lined_img = copy.deepcopy(rm_noi_img)
    for coord in cell_pix_coord_ls:
        cv2.rectangle(cell_lined_img, coord[0], coord[1], (0, 0, 0), 3)
    cell_filename = res_path + file_path + file + '_cell.' + ext
    print(f'{cell_filename} saved')
    cv2.imwrite(cell_filename, cell_lined_img)
    ###
    
    cell_pix_mat_coord_ls = list(zip(cell_pix_coord_ls, cell_mat_coord_ls))  # 셀 픽셀 - 행열 좌표 매핑
    text_coord_ls = text_box['coord']                                        # 텍스트 영역 좌표((좌상단) , (우하단))
    text_img = text_box['img']
    
    text_box_path = res_path+file_path+file+'_text_box.'+ext
    print(f'{text_box_path} saved' )
    cv2.imwrite(text_box_path, text_img)
    
    # 셀 - 텍스트 좌표 매핑
    cell_text_coord_ls = map_cell_text(cell_pix_mat_coord_ls, text_coord_ls)
    
    # 글자 검출
    conf = r'--oem 1 --psm 8'
    sp_path = res_path + file_path + split_path
    cell_text_dict = rec_text(rm_noi_img, cell_text_coord_ls, filename, sp_path, conf)
    print(f'{sp_path} saved')
    
    # 정렬 후 출력
    n_row = len(mod_y_line_ls)
    n_col = len(mod_x_line_ls)
    
    df = result_to_csv(n_row, n_col, cell_text_dict, filename)
    
    csv_path = res_path + file_path + file+'_res.csv'
    df.to_csv(csv_path, encoding='utf8', index=False) 
    print(f'{csv_path} saved')

        
    
if __name__ == '__main__':
    main()