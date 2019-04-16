#针对卷积模型的权重保存结构体

class weights_conv:
    def __init__(self):
        self.encoder_h1=None
        self.encoder_h2=None
        self.encoder_b1 = None
        self.encoder_b2 = None
        self.decoder_h1 = None
        self.decoder_h2 = None
        self.decoder_b1 = None
        self.decoder_b2 = None