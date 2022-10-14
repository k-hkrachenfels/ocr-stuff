import easyocr
from PIL import Image
from PIL.ImageDraw import Draw
from PIL import ImageEnhance


reader = easyocr.Reader(['de','en']) # this needs to run only once to load the model into memory
result = reader.readtext('bilder/patch.jpeg')
#result = reader.readtext('bilder/mit_blitz.jpeg')
print(result)

# for triple in result:
#     points,text,stroke = triple
#     print(f"points={points}, text={text}, stroke={stroke}")



# with Image.open("bilder/mit_blitz.jpeg") as im:
#     im.show()
#     enhancer = ImageEnhance.Sharpness(im)
#     im = enhancer.enhance(4)
#     #enhancer = ImageEnhance.Contrast(im)
#     #im = enhancer.enhance(2)
#     enhancer = ImageEnhance.Brightness(im)
#     im = enhancer.enhance(1.5)
#     enhancer = ImageEnhance.Color(im)
#     im = enhancer.enhance(0.2)
#     draw = Draw(im)
#     for triple in result:
#         points,text,stroke = triple
#         points.append(points[0])
#         draw.text(points[0],text)
#         for index in range(len(points)-1):
#             start=points[index]
#             line=start.copy()
#             end=points[index+1]
#             line.extend(end)
#             print(line)
#             draw.line(line,fill=128,width=3)
#    im.show()
#    im.save("bilder/enhanced.jpeg")       

with Image.open("bilder/mit_blitz.jpeg") as im:
    im.show()
    #im.save("bilder/enhanced.jpeg")

    patch = im.crop([1145, 1747, 1374, 1837])
    patch.save("bilder/patch.jpeg")
