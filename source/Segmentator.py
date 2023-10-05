from source import U2NET
from source import U2NETP
from skimage import io, transform, color
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import cv2

class ImageSegmentator:
    def __init__(self, weight_path, model_name='u2net', device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        if model_name == 'u2net':
            model = U2NET(3, 1)
        elif model_name == 'u2netp':
            model = U2NETP(3, 1)

        # Load trained model 
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.to(device).eval()
        self.model = model
        self.device = device

    def preprocessor(self, img_raw):
        img = RescaleT(320).transf(img_raw)
        img = ToTensorLab(flag=0).transf(img)
        img = torch.tensor(img, dtype=torch.float32).to(self.device)
        img = img.unsqueeze(0)
		
        return img

    def segment(self, ori_img):
        input_img = self.preprocessor(ori_img)
        print("input image shape:", input_img.shape)
        print("input image type:", type(input_img))
        inputs_test = input_img.type(torch.FloatTensor)
        if torch.cuda.is_available():
            input_img = Variable(input_img.cuda())
        else:
            input_img = Variable(input_img)

        d1, d2, d3, d4, d5, d6, d7 = self.model(input_img)
        pred = d1[:, 0, :, :]
        pred = normPRED(pred)

        pred = output_result(ori_img, pred)
        return pred
    
class RescaleT(object):

	def __init__(self,output_size):
		assert isinstance(output_size,(int,tuple))
		self.output_size = output_size

	def transf(self, image):

		h, w = image.shape[:2]

		if isinstance(self.output_size,int):
			if h > w:
				new_h, new_w = self.output_size*h/w,self.output_size
			else:
				new_h, new_w = self.output_size,self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)

		# #resize the image to new_h x new_w and convert image from range [0,255] to [0,1]
		# img = transform.resize(image,(new_h,new_w),mode='constant')
		# lbl = transform.resize(label,(new_h,new_w),mode='constant', order=0, preserve_range=True)

		img = transform.resize(image,(self.output_size,self.output_size),mode='constant')
            
		return img
      
class ToTensorLab(object):
	"""Convert ndarrays in sample to Tensors."""
	def __init__(self,flag=0):
		self.flag = flag

	def transf(self, image):
		# if(np.max(label)<1e-6):
		# 	label = label
		# else:
		# 	label = label/np.max(label)

		# change the color space
		if self.flag == 2: # with rgb and Lab colors
			tmpImg = np.zeros((image.shape[0],image.shape[1],6))
			tmpImgt = np.zeros((image.shape[0],image.shape[1],3))
			if image.shape[2]==1:
				tmpImgt[:,:,0] = image[:,:,0]
				tmpImgt[:,:,1] = image[:,:,0]
				tmpImgt[:,:,2] = image[:,:,0]
			else:
				tmpImgt = image
			tmpImgtl = color.rgb2lab(tmpImgt)

			# nomalize image to range [0,1]
			tmpImg[:,:,0] = (tmpImgt[:,:,0]-np.min(tmpImgt[:,:,0]))/(np.max(tmpImgt[:,:,0])-np.min(tmpImgt[:,:,0]))
			tmpImg[:,:,1] = (tmpImgt[:,:,1]-np.min(tmpImgt[:,:,1]))/(np.max(tmpImgt[:,:,1])-np.min(tmpImgt[:,:,1]))
			tmpImg[:,:,2] = (tmpImgt[:,:,2]-np.min(tmpImgt[:,:,2]))/(np.max(tmpImgt[:,:,2])-np.min(tmpImgt[:,:,2]))
			tmpImg[:,:,3] = (tmpImgtl[:,:,0]-np.min(tmpImgtl[:,:,0]))/(np.max(tmpImgtl[:,:,0])-np.min(tmpImgtl[:,:,0]))
			tmpImg[:,:,4] = (tmpImgtl[:,:,1]-np.min(tmpImgtl[:,:,1]))/(np.max(tmpImgtl[:,:,1])-np.min(tmpImgtl[:,:,1]))
			tmpImg[:,:,5] = (tmpImgtl[:,:,2]-np.min(tmpImgtl[:,:,2]))/(np.max(tmpImgtl[:,:,2])-np.min(tmpImgtl[:,:,2]))

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])
			tmpImg[:,:,3] = (tmpImg[:,:,3]-np.mean(tmpImg[:,:,3]))/np.std(tmpImg[:,:,3])
			tmpImg[:,:,4] = (tmpImg[:,:,4]-np.mean(tmpImg[:,:,4]))/np.std(tmpImg[:,:,4])
			tmpImg[:,:,5] = (tmpImg[:,:,5]-np.mean(tmpImg[:,:,5]))/np.std(tmpImg[:,:,5])

		elif self.flag == 1: #with Lab color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))

			if image.shape[2]==1:
				tmpImg[:,:,0] = image[:,:,0]
				tmpImg[:,:,1] = image[:,:,0]
				tmpImg[:,:,2] = image[:,:,0]
			else:
				tmpImg = image

			tmpImg = color.rgb2lab(tmpImg)

			# tmpImg = tmpImg/(np.max(tmpImg)-np.min(tmpImg))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.min(tmpImg[:,:,0]))/(np.max(tmpImg[:,:,0])-np.min(tmpImg[:,:,0]))
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.min(tmpImg[:,:,1]))/(np.max(tmpImg[:,:,1])-np.min(tmpImg[:,:,1]))
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.min(tmpImg[:,:,2]))/(np.max(tmpImg[:,:,2])-np.min(tmpImg[:,:,2]))

			tmpImg[:,:,0] = (tmpImg[:,:,0]-np.mean(tmpImg[:,:,0]))/np.std(tmpImg[:,:,0])
			tmpImg[:,:,1] = (tmpImg[:,:,1]-np.mean(tmpImg[:,:,1]))/np.std(tmpImg[:,:,1])
			tmpImg[:,:,2] = (tmpImg[:,:,2]-np.mean(tmpImg[:,:,2]))/np.std(tmpImg[:,:,2])

		else: # with rgb color
			tmpImg = np.zeros((image.shape[0],image.shape[1],3))
			image = image/np.max(image)
			if image.shape[2]==1:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,2] = (image[:,:,0]-0.485)/0.229
			else:
				tmpImg[:,:,0] = (image[:,:,0]-0.485)/0.229
				tmpImg[:,:,1] = (image[:,:,1]-0.456)/0.224
				tmpImg[:,:,2] = (image[:,:,2]-0.406)/0.225

		tmpImg = tmpImg.transpose((2, 0, 1))

		return tmpImg
	
def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)
    return dn

def output_result(ori_img, result_img):
	H, W = ori_img.shape[0], ori_img.shape[1]
	predict = result_img.squeeze()
	predict_np = predict.cpu().data.numpy() * 255
	im = Image.fromarray(predict_np).convert('RGB')
	imo = im.resize((W, H),resample=Image.BILINEAR)
	# output_img = cv2.resize(predict_np, (W, H))
	output_img = np.array(imo)
	return output_img
    