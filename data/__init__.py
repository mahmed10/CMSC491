from data import cityscapes, synthia
from torch.utils.data import DataLoader, ConcatDataset

def setup_loaders(dataset, path_list, batch_size):
	if(dataset == 'cityscapes'):
		data = cityscapes.CityScapes(path_list)
		data_loader = DataLoader(data, batch_size=batch_size)

	if(dataset == 'synthia'):
		data = synthia.Synthia(path_list)
		data_loader = DataLoader(data, batch_size=batch_size)
	return data_loader