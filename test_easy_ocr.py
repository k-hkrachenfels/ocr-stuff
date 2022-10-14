import easyocr
from PIL import Image
from PIL.ImageDraw import Draw
from PIL import ImageEnhance
from typing import List



def pretty_print_ocr(ocr_result_list: List):
    for points,text,score in ocr_result_list:
        print(f"text={text}, score:{score}, rect={points}")

def ocr_image_from_file(image_name: str):
    reader = easyocr.Reader(['de','en']) # this needs to run only once to load the model into memory
    ocr_result = reader.readtext(image_name)
    return ocr_result

def draw_boxes_into_image(in_file_name: str, out_file_name: str, ocr_result_list: List):
    with Image.open(in_file_name) as im:
        im.show()
        # enhancer = ImageEnhance.Sharpness(im)
        # im = enhancer.enhance(4)
        # #enhancer = ImageEnhance.Contrast(im)
        # #im = enhancer.enhance(2)
        # enhancer = ImageEnhance.Brightness(im)
        # im = enhancer.enhance(1.5)
        # enhancer = ImageEnhance.Color(im)
        # im = enhancer.enhance(0.2)
        draw = Draw(im)
        for points,text,stroke in ocr_result_list:
            # append the first point at the end of the sequence of points for polygon drawing
            points.append(points[0])
            draw.text(points[0],text)

            for index in range(len(points)-1):
                start=points[index]
                line=start.copy()
                end=points[index+1]
                line.extend(end)
                draw.line(line,fill=128,width=3)
        #im.show()
        im.save(out_file_name)      

def extract_image_patch(coords: List[int], in_file_name: str, out_file_name: str):
    with Image.open(in_file_name) as im:
        #im.show()
        patch = im.crop(coords)
        patch.save(out_file_name)

input_file="bilder/mit_blitz.jpeg"
output_file="bilder/mit_blitz_and_ocr_results.jpeg"
ocr_result_list = ocr_image_from_file(input_file)
pretty_print_ocr(ocr_result_list)
draw_boxes_into_image(input_file, output_file, ocr_result_list)
extract_image_patch([1145, 1747, 1374, 1837], "bilder/mit_blitz.jpeg", "bilder/patch.jpeg")