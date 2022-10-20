from PIL import Image
import yaml
from collections import defaultdict


def extract_digits( in_file_name: str, output_folder: str):
    with open('config/config.yaml') as f:
        config_dict = yaml.safe_load(f)
        sensor_dict = config_dict['sensors']
        print(sensor_dict)

        # apply calibration
        x = config_dict['calibration']['x']
        y = config_dict['calibration']['y']
        calibrated_sensors = defaultdict(list)
        for sensor_name,sensor_boxes in sensor_dict.items():
            calibrated_boxes=[]
            for box in sensor_boxes:
                x1,y1,x2,y2 = box['digit_box']
                cal_box = [x1+x,y1+y,x2+x,y2+y]
                calibrated_sensors[sensor_name].append( {'digit_box':cal_box})   
        print(f"calibrated sensors: {calibrated_sensors}")


        sensor_dict = calibrated_sensors
        with Image.open(in_file_name) as im:
            im = im.rotate(180)
            im.save("bilder/rotated.png")
            global_index=0
            for sensor_key in sensor_dict.keys():
                for i, digit in enumerate(sensor_dict[sensor_key]):
                    digit = im.crop(digit['digit_box'])
                    #digit.save(f"digits/{sensor_key}_{i}.png")
                    digit.save(f"{output_folder}/img_{global_index}.png")
                    global_index+=1
            
if __name__ == "__main__":
    extract_digits("bilder/patch3.png", "digits")
