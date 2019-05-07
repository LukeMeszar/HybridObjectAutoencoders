import urllib.request
import os
import shutil

try:
    os.mkdir("data")
except:
    print("/data already exists")

print("Downloading Trained Models...")
data_getter = urllib.request.URLopener()
data_getter.retrieve("https://grantbaker.keybase.pub/data/background_filtered/test_imagefolder.zip", "test_imagefolder.zip")
data_getter.retrieve("https://grantbaker.keybase.pub/data/background_filtered/train_imagefolder.zip", "train_imagefolder.zip")
data_getter.retrieve("https://grantbaker.keybase.pub/data/animals/animals.zip", "animals.zip")
shutil.move("test_imagefolder.zip", "data/test_imagefolder.zip")
shutil.move("train_imagefolder.zip", "data/train_imagefolder.zip")
shutil.move("animals.zip", "data/animals.zip")

print("Done")
