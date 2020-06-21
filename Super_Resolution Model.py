class ModelSR():


    def __init__(self, image_size):
        self.image_size = image_size

    def Res_block(self, ip):
        init = ip
        x = Convolution2D(64, (3, 3), activation='linear', padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Convolution2D(64, (3, 3), activation='linear', padding='same')(x)
        x = BatchNormalization()(x)
        m = concatenate(axis=3)([x, init])
        return m

    def SR_ResBlock(self):
        inp = Input((None, None,3))
        C1 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(inp)
        C2 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C1) 
        C3 = Conv2D(64, kernel_size=(3,3), padding='same', activation='relu')(C2) 
        x = self.Res_block(C3)     
        for i in range(2):
                x = self.Res_block(x)
        Cout = Conv2D(31, kernel_size=(3,3), padding='same', activation='relu')(x)
        model = Model(inp, Cout)
        return model
