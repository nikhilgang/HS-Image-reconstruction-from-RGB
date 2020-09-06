import keras
from keras.layers import (Input,Activation, Conv2D, Dropout, Convolution2D,UpSampling2D)
from keras.layers import (Activation, Conv3D, Dense, Dropout, Flatten,Conv2DTranspose,
                  Add,MaxPooling2D,MaxPooling3D, Input, concatenate,BatchNormalization)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import multiply
from keras.losses import mean_absolute_error
from keras.models import Sequential,load_model,Model
from keras.optimizers import Adam
import os, math
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import cv2
import argparse
import numpy as np
import tensorflow as tf
from skimage import io, img_as_uint, img_as_ubyte
from scipy.io import loadmat
# import matplotlib
# # matplotlib.use('AGG')
# import matplotlib.pyplot as plt


class model_rgb2hs():

        def __init__(self, image_size):
                self.image_size = image_size

        def residual_block(self, ip):
                init = ip
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(ip)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
                x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x)
                x = BatchNormalization()(x)
                m = concatenate(axis=3)([x, init])
                return m

        def SR_ResNet(self):
                inp = Input((None, None,3))
                C1 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inp)
                C2 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C1) 
                C3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C2) 
                x = self.residual_block(C3)     
                for i in range(2):
                        x = self.residual_block(x)
                Cout = Conv2D(31, kernel_size=(3,3), padding='same', activation='relu')(x)
                model = Model(inp, Cout)
                return model


class keras_utils():
        
        def mrae_loss(self,y_pred, y_true):
                import keras.backend as K
                difference = K.abs(y_true - y_pred)/y_true
                mrae = K.mean(difference)
                return mrae

        def load_rgb2hs_model(self,old_model_path=None):
                print("[INFO] Training Resumes with........", os.path.split(old_model_path)[0])
                model = load_model(old_model_path)
                model.summary()
                return model

        def start_train(self,model, model_name, X_data, Y_data, X_val, Y_val,epochs, batch_size, lr, model_save_dir, version, initial_epoch=0):
                model.compile(loss='mean_absolute_error', optimizer=Adam(lr=lr), metrics=['mae','mse'])
                checkpoint = keras.callbacks.ModelCheckpoint(model_save_dir+'/model_'+str(model_name)+'_'+version+'_{epoch:02d}-{loss:.4f}.h5',monitor='loss',
                                                                verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=50)
                csv_logger = keras.callbacks.CSVLogger(model_save_dir + "/tr_-v"+version+".csv", separator=',', append=True)
                # history = model.fit(x=X_data, y=Y_data, batch_size=batch_size, epochs=epochs, verbose=1, initial_epoch=initial_epoch,
                #                                       validation_split=0.2,callbacks=[checkpoint, csv_logger], shuffle=False)
                history = model.fit(x=X_data, y=Y_data, batch_size=batch_size, epochs=epochs, verbose=1, initial_epoch=initial_epoch, validation_split=0.2, 
                                                                        validation_data=(X_val, Y_val), callbacks=[checkpoint, csv_logger], shuffle=True)
                return history

        def plot_history_regression(self,history,version,epochs,model_save_dir,model_name):
                plt.plot(history.history['loss'], 'b', label='Training loss')
                plt.plot(history.history['val_loss'], 'r', label='Validation loss')

                plt.title('Training and Validation Loss -v'+str(model_name)+str(version))
                plt.legend()
                plt.savefig(os.path.join(model_save_dir, str(model_name)+'_train_loss_'+str(version)+'_'+str(epochs)+'.png')) 
                print("Saved Training history as ....",str(model_name)+'_train_loss_'+str(version)+'_'+str(epochs)+'.png')

class common_utils():

        def append_image(self,data_dir,data_save_path,image_size=None,resize=False):
                img_list = []
                count = 0
                for img in sorted(os.listdir(data_dir)):
                        img_path = os.path.join(data_dir, img)

                        image = cv2.imread(img_path)
                        img_list.append(image)
                        count += 1
                        print(np.array(img_list).shape,end='\r')
                try:
                        np.save(data_save_path, np.array(img_list))
                        print('\nsaved....... ', os.path.split(data_save_path)[1], np.array(img_list).shape)
                except :
                        print("[WARNING]..Exception Occured ......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print('[INFO] To save image array as numpy file provide location with file name')

                return np.array(img_list)

        def create_patch(self, image, patch_dimension,overlap=0):
                height = image.shape[0]
                width = image.shape[1]
                # print(height, width)
                patch_list = []
                i = 0
                cnt = 0
                while (i<height):
                        j=0
                        while (j<width):
                                if i+patch_dimension <= height-1 and j+patch_dimension <= width-1:
                                        rs=i
                                        re = i+patch_dimension
                                        cs = j
                                        ce = j+patch_dimension
                                        # print ('if-1',i,j)
                                if i+patch_dimension >= height and j+patch_dimension <=width-1:
                                        rs = height-(patch_dimension)
                                        re = height
                                        cs = j
                                        ce = j+patch_dimension
                                        # print ('if-2',i,j)
                                if i+patch_dimension <= height-1 and j+patch_dimension >=width:
                                        rs = i
                                        re = i+patch_dimension
                                        cs = width - (patch_dimension)
                                        ce = width
                                        # print ('if-3',i,j)
                                        #print j
                                if i+patch_dimension >= height and j+patch_dimension >=width:
                                        rs = height-(patch_dimension)
                                        re = height
                                        cs = width - (patch_dimension)
                                        ce = width
                                        # print ('if-4',i,j)
                                # print(rs,":",re,",",cs,":",ce)
                                # j+=1
                                cropimage = image[rs:re, cs:ce]
                                # cv2.imwrite(str(cnt)+'.png', cropimage)
                                cnt += 1
                                # print(cnt)
                                patch_list.append(cropimage)
                                j=j+(200-overlap)
                        i=i+(200-overlap)
                                # pix = cropimage
                # print(np.array(patch_list).shape)                     
                return np.array(patch_list)


        def create_patch_combined(self, data, patch_dimension,overlap,file_save_path=None):
                patch_list1 = []
                temp_patch = np.array(0)
                cnt = 0
                for i in range(data.shape[0]):
                        img_patch = self.create_patch(data[i], patch_dimension,overlap)
                        # print(img_patch.shape)
                        if cnt == 0:
                                temp_patch = img_patch
                        elif cnt > 0:
                                temp_patch = np.concatenate((temp_patch, img_patch), axis=0)
                                print(i, temp_patch.shape, img_patch.shape, end='\r')
                        cnt += 1
                        patch_list1.append(img_patch)
                        
                        if i >0 and i % 50 == 0:
                                if data.shape[3] == 31:                         
                                        location = os.path.split(os.path.split(file_save_path)[0])[0]+'/'+'hs_patch'
                                        if not os.path.exists(location):
                                                os.makedirs(location)
                                                os.chmod(location, 0o777)
                                        np.save(location+'/'+'hs_patch20_'+str(i).zfill(3)+'.npy', 
                                                np.array(temp_patch))
                                        print('\n saved file', np.array(temp_patch).shape,i)
                                        temp_patch = np.array(0)
                                        cnt = 0         
                if data.shape[3] == 31:
                        location = os.path.split(os.path.split(file_save_path)[0])[0]+'/'+'hs_patch'
                        np.save(location+'/'+'hs_patch20_'+str(i).zfill(3)+'.npy', 
                                np.array(temp_patch))
                        print('\n saved file', np.array(temp_patch).shape,i)
                        count = 0
                        temp_patch = np.array(0)
                        for npFile in sorted(os.listdir(location)):
                                x = np.load(os.path.join(location, npFile))
                                if count == 0:
                                        temp_patch = x
                                        print(npFile, temp_patch.shape, x.shape)
                                if count > 0:
                                        temp_patch = np.concatenate(axis=3)((temp_patch, x), axis=0)
                                        print(npFile, temp_patch.shape, x.shape, end='\r')
                                count += 1
                
                try:
                        np.save(file_save_path, np.array(temp_patch))
                        print('\n saved file.......' ,os.path.split(file_save_path)[1] ,np.array(temp_patch).shape)
                except FileNotFoundError as e:
                        print("[Error].............. provided path does not exists")
                except Exception as e:
                        print("[WARNING]..Exception Occured ......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print('[INFO] To save patch numpy file provide file_save_path with file name')

                print('Coverted Dataset into Patches ....................',np.array(temp_patch).shape )
                
                return np.array(temp_patch)


        def resize_npFile(self,data, image_size, file_save_path=None):  
                print("Resizing the input data .................",data.shape)
                resize_np = np.zeros((data.shape[0],image_size, image_size, data.shape[3]))
                
                for j in range(data.shape[0]):
                        for i in range(data.shape[3]):                  
                                # cv2.imwrite('org.png',img_as_ubyte(data[j][:,:,i]))
                                temp = cv2.resize(data[j][:,:,i],(image_size, image_size), interpolation=cv2.INTER_AREA)                        # cv2.imwrite('1.png', img_as_ubyte(temp))
                                resize_np[j][:,:,i] = temp
                                # print(data[j].shape, data[j][:,:,i].shape, temp.shape, resize_np[j].shape)                    
                try:
                        np.save(file_save_path, resize_np)
                        print('Successfully Saved .....', file_save_path, resize_np.shape)      
                except Exception as e:
                        print("[WARNING]..Exception Occured ......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print('[INFO] To save resized numpy file provide file_save_path with file name')
                return resize_np

        def normalize_data(self, data):
                print("Normalizing Data ...........................")
                print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
                # temp = img_as_ubyte(data)
                # print(temp.shape, temp.dtype, "min:",temp.min(), "max:",temp.max())
                data = data/data.max()
                print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
                return data     

        def load_Data(self,data_dir, data_save_dir, file_save_name):
                if os.path.exists(data_save_dir+'/'+file_save_name+'.npy'):
                        train_data = np.load(data_save_dir+'/'+file_save_name+'.npy')
                        print('Data loaded ........................',file_save_name+'.npy', train_data.shape)
                else:
                        train_data = self.append_image(data_dir, data_save_dir, file_save_name) 
                return train_data






        def load_validationData(self,data_dir):
                # val_img = np.zeros(image_size, image_size, nchannel)
                count=0
                im_list = []
                for img in sorted(os.listdir(data_dir)):
                        img_path = os.path.join(data_dir, img)
                        ##print("The Y directory is",img_path)

                        #image = cv2.imread(img_path)
                        mat = loadmat(img_path)
                        #print(mat['cube'])
                        data = np.array(mat['cube'])
                        #print(data)
                        ##print("*******")
                        ##print("len of data is: ",data.size)
                        ##print("******")
                        #print(image)
                        image=np.asarray(data)
                        #print(image)
                        ##print("shape is ############",image.shape)

                        if (count==0):
                                tempImg = image
                                print("going")
                                print(image.shape, tempImg.shape)
                                
                        if count >0:
                                tempImg = np.concatenate((tempImg, image),axis=2)
                                print("count>0")
                                print(image.shape, tempImg.shape)
                        count += 1
                        if tempImg.shape[2] == 31:
                                count = 0
                                im_list.append(tempImg)
                                tempImg = []
                                print("loop3")
                                print(np.array(im_list).shape)
                                
                                
                                
                                
                        
                print("the final shape of Y in this function is",np.array(im_list).shape)
                
                return np.asarray(im_list)







        
        def load_groundTruth(self,data_dir, data_save_dir, file_save_name):
                if os.path.exists(data_save_dir+'/'+file_save_name+'.npy'):
                        final_numpyData = np.load(data_save_dir+'/'+file_save_name+'.npy')
                else:
                        count = 0
                        final_numpyData = np.array(0)
                        for matfile in sorted(os.listdir(data_dir)):
                                npFile='C:\\Users\\user\\nik\\dataset\\NTIRE2020_Train_Spectral'
                                matFilePath = os.path.join(matfile, npFile)
                                npTemp = loadmat(npFilePath)
                                if count == 0:
                                        final_numpyData = npTemp
                                        print(npFile, final_numpyData.shape, npTemp.shape)
                                        print(matFilePath, final_numpyData.shape, npTemp.shape)
                                        print(final_numpyData.shape, npTemp.shape)
                                elif count > 0:
                                        final_numpyData = np.concatenate((final_numpyData, npTemp), axis=3)
                                        print(npFile,  npTemp.shape, final_numpyData.shape, end='\r')
                                        print(matFilePath,  npTemp.shape, final_numpyData.shape, end='\r')
                                count += 1      
                        try:
                                np.save(data_save_dir+'/'+file_save_name+'.npy',final_numpyData)
                                print('successfully saved',data_save_dir+'/'+file_save_name+'.npy',end='\n')            
                        except Exception as e:
                                print("[WARNING]..Exception Occured ......!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                                print('[INFO] To save consolidated numpy file provide file_save_path with file name')
                
                print('Ground Truth data loaded...........', final_numpyData.shape)
                return final_numpyData
