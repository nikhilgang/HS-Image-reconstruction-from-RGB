def normalize_data(self, data):
                print("Normalizing Data ...........................")
                print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
                # temp = img_as_ubyte(data)
                # print(temp.shape, temp.dtype, "min:",temp.min(), "max:",temp.max())
                data = data/data.max()
                print(data.shape, data.dtype, "min:",data.min(), "max:",data.max())
                return data
