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
