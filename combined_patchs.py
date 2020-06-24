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
