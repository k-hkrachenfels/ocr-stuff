from PIL import Image
import yaml

digit_size = None
with open('config/config.yaml') as f:
    config_dict = yaml.safe_load(f)

    if config_dict is None:
        raise Exception("empty config")

    for sensor_key in config_dict.keys():
        for i, digit in enumerate(config_dict[sensor_key]):
            left,top,right,bottom = digit['digit_box']
            size = (bottom-top, right-left)
            if digit_size:
                if size == digit_size:
                    continue
                else:
                    message = f"size for {sensor_key} digit {i} does not match expected size {digit_size} got {size} instead."
                    raise Exception(message)
            else:
                digit_size=size

    print(f"Congrats. All boxes have the same size:  {digit_size}")


