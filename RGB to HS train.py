from models_utils import *
import numpy as np
import os, stat, sys

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--version", type=str, default=None, help="to set version of model saved")
	parser.add_argument("--gpu", type=str, default=1,  help="to set number of gpus")
	parser.add_argument("--mode", type=str, default='train' ,choices=['train','test','check'])
	parser.add_argument("--epochs", type=int, default=10, help="to set number of epochs")
	parser.add_argument("--batch_size", type=int, default=64, help="number of input channels")
	args = parser.parse_args()
	return args

def Model_L1(image_size):
        model = Sequential()
        model.add(Convolution2D(32, kernel_size=(3, 3), padding='same',
	                            input_shape=(image_size,image_size,3)))
        model.add(Activation('relu'))
        model.add(Convolution2D(32, kernel_size=(3, 3),padding='same'))
        model.add(Activation('relu'))
        # model.add(Dropout(0.25))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        model.add(Convolution2D(64, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))
        # model.add(Dropout(0.25))

        model.add(Convolution2D(31, kernel_size=(3, 3), padding='same'))
        model.add(Activation('relu'))

        return model

def normalize_data(data):
	print("Normalizing Data ...........................")
	print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
	temp = img_as_ubyte(data)
	print(temp.shape, temp.dtype, "min:",temp.min(), "max:",temp.max())
	data = temp/temp.max()
	print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
	return data	

def resize_npFile(data, image_size, file_save_path=None, save=False):	
	print("Resizing the input data .................",data.shape)
	resize_np = np.zeros((data.shape[0],image_size, image_size, data.shape[3]))
	
	for j in range(data.shape[0]):
		for i in range(data.shape[3]):			
			# cv2.imwrite('org.png',img_as_ubyte(data[j][:,:,i]))
			temp = cv2.resize(data[j][:,:,i],(image_size, image_size), interpolation=cv2.INTER_AREA)	
			resize_np[j][:,:,i] = temp
			# print(data[j].shape, data[j][:,:,i].shape, temp.shape, resize_np[j].shape)			
	if save == True:
		np.save(file_save_path, resize_np)
		print('Successfully Saved .....', file_save_path, resize_np.shape)	
	return resize_np


def main():
	parser = argparse.ArgumentParser()
	args = get_args()
	os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
	print("[INFO] Using GPU {} ...".format(args.gpu))
	nGPU = math.ceil((len(str(args.gpu))+1)/2) 

	pwd = os.getcwd()
	image_size = 256

	if args.mode ==  'train':
		#if args.version is None:
		#	parser.error('Training reequired version. Please provide a version for saving model')

		
		train_data_dir = os.path.join(pwd,'dataset','ntire_dataset','train_numpy','clean_rgb')
		trainFile_save_name = 'clean_rgb_010'	

		print('numpy file for input data patches already exists. Loading File..................')
		X_data = np.load(train_data_dir+'/'+trainFile_save_name+'.npy')
		print('Data loaded ........................',trainFile_save_name+'.npy', X_data.shape)	
		
		
		# print(X_data.shape, X_data.dtype, X_data.min(), X_data.max())
		

		gt_comb_dir = os.path.join(pwd, 'dataset','ntire_dataset','train_numpy','hs_combined')
		gtFile_save_name = 'hs_complete_010'
		Y_data = np.load(gt_comb_dir+'/'+gtFile_save_name+'.npy')
		print('Data loaded ........................',gtFile_save_name+'.npy', Y_data.shape)		
		print(Y_data.shape, Y_data.dtype, Y_data.min(), Y_data.max())	
		
		X_data = normalize_data(X_data)
		print(X_data.shape, X_data.dtype, X_data.min(), X_data.max())

		X_data = resize_npFile(X_data, image_size, file_save_path=None, save=False)
		Y_data = resize_npFile(Y_data, image_size, file_save_path=None, save=False)

		from sklearn.model_selection import train_test_split
		X_train,  X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size=0.2)		
		
		print("Training data and Ground truth shape.................",X_train.shape,Y_train.shape)
		print('Validation Split Completed...........................', X_val.shape, Y_val.shape)

		model_save_dir = os.path.join(pwd,'models_rgb2hs'+'_'+str(args.version))

                
		#model_to_train = old_rgb2hs(image_size)
		model_to_train = Model_L1(image_size)
		
		
		model_to_train.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0001), metrics=['mae','mse'])
		#checkpoint = keras.callbacks.ModelCheckpoint(model_save_dir+'/model_'+args.version+'_{epoch:02d}-{loss:.4f}.h5',monitor='loss',
		#						verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=50)
		#history = model_to_train.fit(x=X_data, y=Y_data, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=0.2, 
		#							validation_data=(X_val, Y_val),callbacks=[checkpoint], shuffle=True)
		history = model_to_train.fit(x=X_data, y=Y_data, batch_size=args.batch_size, epochs=args.epochs, verbose=1, validation_split=0.2, 
									validation_data=(X_val, Y_val), shuffle=True)


		plt.plot(history.history['loss'], 'b', label='Training loss')
		#plt.plot(history.history['mean_absolute_error'], 'r', label='mean_absolute_error loss')
		#plt.plot(history.history['mean_squared_error'], 'g', label='mean_squared_error loss')
		plt.plot(history.history['mae'], 'r', label='mean_absolute_error loss')
		plt.plot(history.history['mse'], 'g', label='mean_squared_error loss')

		plt.title('Training loss -v'+str(args.version))
		plt.legend()
		plt.savefig(os.path.join(model_save_dir, 'model_train_loss_'+str(args.version)+'.png'))



if __name__ == '__main__':
	main()
