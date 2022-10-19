from PIL import Image
import yaml

with open('config/config.yaml') as f:
    config_dict = yaml.safe_load(f)
    print(config_dict)

in_file_name = "bilder/patch3.png"
with Image.open(in_file_name) as im:
    im = im.rotate(180)
    im.save("bilder/rotated.png")
    global_index=27
    for sensor_key in config_dict.keys():
        for i, digit in enumerate(config_dict[sensor_key]):
            digit = im.crop(digit['digit_box'])
            #digit.save(f"digits/{sensor_key}_{i}.png")
            digit.save(f"digits/img_{global_index}.png")
            global_index+=1
            