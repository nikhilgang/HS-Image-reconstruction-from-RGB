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
