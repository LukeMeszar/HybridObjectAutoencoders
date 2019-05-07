import urllib.request

print("Downloading Trained Models...")
model_getter = urllib.request.URLopener()
model_getter.retrieve("https://grantbaker.keybase.pub/saved_models/stl_1.pth", "model_stl.pth")
model_getter.retrieve("https://grantbaker.keybase.pub/saved_models/cifar-v2-model-3.pth", "model_cifar_v1.pth")
model_getter.retrieve("https://grantbaker.keybase.pub/saved_models/cifar-v2-model-4.pth", "model_cifar_v2.pth")
model_getter.retrieve("https://grantbaker.keybase.pub/saved_models/stl_unet_1.pth", "model_stl_u.pth")

print("Done")
