import os

def model_save(model, optimizer, path):
	torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
         }, os.path.join(path,"model.pth"))
	return("model saved to {}".format(os.path.join(path,"model.pth")))

def model_load(model_class, opt_class, path):
	model = model_class
	optimizer = opt_class

	checkpoint = torch.load(os.path.join(path,"model.pth"))
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	return(model, optimizer)

def model_save_simple(model, path):
	torch.save(model, os.path.join(path,"model.pth"))
	return("model saved to {}".format(os.path.join(path,"model.pth")))

def model_load_simple(path):
	model = torch.load(os.path.join(path,"model.pth")
	return(model)
