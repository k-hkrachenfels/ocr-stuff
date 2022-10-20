import requests
import urllib.request
from PIL import Image
import time

from torchvision import transforms
import torch
from train_digit_recognition import get_test_kwargs, \
    get_test_transforms, init_device, get_args, \
    get_test_kwargs, get_test_transforms, get_model
from datasets import DigitDataset
from extract_digits import extract_digits

capture_url = "http://192.168.1.149/capture"
image_url = "http://192.168.1.149/saved-photo"
file_name = "bilder/patch3.png"
sleep_time = 30


def request_data():
    print(f"requesting capture from {capture_url}")
    ret = requests.get(capture_url)
    print(f"ret =  {ret}")


    print(f"sleeping {sleep_time} seconds")
    time.sleep(sleep_time)

    print(f"requesting image from {image_url} and writing to {file_name}")
    urllib.request.urlretrieve( image_url, file_name )

    #img = Image.open(file_name)
    print("capture done. Next step: Pattern recognition")

    extract_digits(file_name, "apply")



def apply_model():
    args = get_args() 
    device = init_device(args)
    test_kwargs = get_test_kwargs(args, device)
    test_transforms = get_test_transforms()
    dataset_test = DigitDataset('./apply', train=False, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)
    model = get_model(device)
    model.load_state_dict(torch.load("./solar_controller_digit_model.pth"))
    model.to(device)
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True) 
            print(pred)

# connect to edge device and request and preprocess data and write to apply folder
# request_data()

# apply the model to the data in the apply folder
apply_model()

