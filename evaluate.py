import torch
import numpy as np
import numpy
from tqdm import tqdm
from domain_adaptation.utils.func import per_class_iu, fast_hist
import matplotlib.patches as mpatches
from torchvision import transforms
import matplotlib.pyplot as plt
SMOOTH = 1e-6
def iou_calculation(batch, model, device):
	X, labels = batch
	X, labels = X.to(device), labels.to(device)
	pred_main = model(X)[1]
	output = pred_main.cpu().data[0].numpy()
	output = output.transpose(1, 2, 0)
	pred = np.argmax(output, axis=2)
	outputs = torch.tensor(pred.reshape((1, pred.shape[0], pred.shape[1]))).cpu()
	labels = torch.tensor(labels).cpu()
	intersection = (outputs & labels).float().sum((1, 2)) 
	union = (outputs | labels).float().sum((1, 2)) 
	iou = (intersection + SMOOTH) / (union + SMOOTH)
	return iou

def per_class_iou_calculataion(net, target_loader, device):
	targetloader_iter = enumerate(target_loader)
	hist = np.zeros((20,20))
	for i_iter in tqdm(range(1000)):
		_, batch = targetloader_iter.__next__()
		image, label = batch
		image, label = image.to(device), label
		with torch.no_grad():
			pred_main = net(image)[1]
			output = pred_main.cpu().data[0].numpy()
			output = output.transpose(1, 2, 0)
			output = np.argmax(output, axis=2)
		label = label.numpy()[0]
		hist += fast_hist(label.flatten(), output.flatten(), 20)
	inters_over_union_classes = per_class_iu(hist)
	return inters_over_union_classes


def get_activation(name):
	def hook(model, input, output):
		activation[name] = output.detach()
	return hook

def figure_draw(batch, net, device, figname):
	img, mask= batch
	X, y = img.to(device), mask.to(device)
	pred_main = net(X)[1]
	output = pred_main.cpu().data[0].numpy()
	output = output.transpose(1, 2, 0)
	pred = np.argmax(output, axis=2)

	plt.figure()
	plt.title('Target Image')
	trans = transforms.ToPILImage()
	fig = plt.imshow(trans(img[0]))
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

	plt.figure()
	plt.title('Ground Truth')
	values = np.arange(0,20)
	im = plt.imshow(mask[0], cmap= 'tab20')
	# get the colors of the values, according to the 
	# colormap used by imshow
	colors = [ im.cmap(im.norm(value)) for value in values]
	# create a patch (proxy artist) for every color 
	patches = [ mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i]) ) for i in range(len(values)) ]
	im.axes.get_xaxis().set_visible(False)
	im.axes.get_yaxis().set_visible(False)
	# put those patched as legend-handles into the legend
	# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )


	plt.figure()
	plt.title(figname)
	values = np.arange(0,20)
	im = plt.imshow(pred, cmap= 'tab20')
	# get the colors of the values, according to the 
	# colormap used by imshow
	colors = [ im.cmap(im.norm(value)) for value in values]
	# create a patch (proxy artist) for every color 
	patches = [ mpatches.Patch(color=colors[i], label="{l}".format(l=values[i]) ) for i in range(len(values)) ]
	im.axes.get_xaxis().set_visible(False)
	im.axes.get_yaxis().set_visible(False)
	# put those patched as legend-handles into the legend
	# plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

	plt.show()