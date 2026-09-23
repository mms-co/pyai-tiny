def pre_process(img):
	batch = img.shape[0]
	return img.reshape(batch, -1)