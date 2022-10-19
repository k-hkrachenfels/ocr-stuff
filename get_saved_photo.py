import requests
import urllib.request
from PIL import Image
import time

capture_url = "http://192.168.1.149/capture"
image_url = "http://192.168.1.149/saved-photo"
file_name = "bilder/patch3.png"
sleep_time = 30

print(f"requesting capture from {capture_url}")
ret = requests.get(capture_url)
print(f"ret =  {ret}")


print(f"sleeping {sleep_time} seconds")
time.sleep(sleep_time)

print(f"requesting image from {image_url} and writing to {file_name}")
urllib.request.urlretrieve( image_url, file_name )

img = Image.open(file_name)
print("done")
#img.show()



