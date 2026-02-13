class Config:
    def __init__(self, backbone, class_num, k_shot, n_query):
        self.class_num = class_num
        self.backbone = backbone
        self.k_shot = k_shot
        self.n_query = n_query
        if 'RelationNet' in self.backbone:
            self.relation_dim = 8

    def return_config(self):
        config = []
        header_config = []
        if 'resnet10_LAT_backbone' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56
                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # 利用第一层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]), #54*64*56*56

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]

        elif 'resnet10_LAT_backbone_out0' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # 利用第0层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*64*56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]

        elif 'resnet10_LAT_backbone_out1' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # 利用第1层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*128*28*28

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),


                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]
        elif 'resnet10_LAT_backbone_out2' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # 利用第2层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*128*28*28

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]
        elif 'resnet10_LAT_backbone_out3' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # 利用第3层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*256*14*14

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]
        elif 'resnet10_LAT_backbone_out4' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),
                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),

                # 利用第4层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*512*7*7

                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]

        elif 'resnet10_LAT_backbone_conv0' == self.backbone:
            config = [
                # ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d  84
                ('conv2d_in', [64, 3, 7, 7, 2, 3]),  # out 112
                ('bn', [64]),
                ('relu', [True]),

                # 利用第4层残差链接
                ('output_layer', [self.class_num, self.k_shot, self.n_query]),  # 54*512*7*7

                # k s p
                ('max_pool2d_r', [3, 2, 1]),  # 56*56

                # layer1
                ('conv2d_in', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),
                ('relu', [True]),
                ('conv2d', [64, 64, 3, 3, 1, 1]),  # out 56
                ('bn', [64]),

                ('res_add', [64]),
                ('relu_r', [True]),

                # layer2
                ('conv2d_in', [128, 64, 3, 3, 2, 1]),  # out 28
                ('bn', [128]),
                ('relu', [True]),
                ('conv2d', [128, 128, 3, 3, 1, 1]),  # out 28
                ('bn', [128]),

                ('conv_down', [128, 64, 1, 1, 2, 0]),  # 28
                ('bn_down', [128]),

                ('res_add', [128]),
                ('relu_r', [True]),

                # layer3
                ('conv2d_in', [256, 128, 3, 3, 2, 1]),  # out 14
                ('bn', [256]),
                ('relu', [True]),
                ('conv2d', [256, 256, 3, 3, 1, 1]),  # out 14
                ('bn', [256]),

                ('conv_down', [256, 128, 1, 1, 2, 0]),  # 14
                ('bn_down', [256]),

                ('res_add', [256]),
                ('relu_r', [True]),

                # layer4
                ('conv2d_in', [512, 256, 3, 3, 2, 1]),  # 7
                ('bn', [512]),
                ('relu', [True]),
                ('conv2d', [512, 512, 3, 3, 1, 1]),  # out 7
                ('bn', [512]),

                ('conv_down', [512, 256, 1, 1, 2, 0]),
                ('bn_down', [512]),

                ('res_add', [512]),
                ('relu_r', [True]),


                ('mean', []),
                # ('flatten', []),
                # ('linear', [self.class_num, 512])

            ]

        elif 'resnet10_LAT_out0' == self.backbone:
            config = [
                # ('reshape', [-1]),
                ('conv2d', [32,64,3,3,2,1]), # out 28
                ('bn', [32]),
                ('relu', [True]),

                ('conv2d', [16, 32, 3, 3, 2, 1]), # out 14
                ('bn', [16]),
                ('relu', [True]),

                ('conv2d', [8, 16, 3, 3, 2, 1]),  # out 7
                ('bn', [8]),
                ('relu', [True]),

                ('conv2d', [4, 8, 3, 3, 2, 1]),  # out 4
                ('bn', [4]),
                ('relu', [True]),

                ('conv2d', [2, 4, 3, 3, 2, 1]),  # out 2
                ('bn', [2]),
                ('relu', [True]),

                ('avg_pool2d', [2,2,0]),  # out 45*8*1*1

                ('flatten_w',[]),
                # ('linear', [64*3*7*7,512*self.class_num*(self.k_shot+self.n_query)*7*7]),
                ('linear_re', [64 * 3 * 7 * 7, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64 , 3 , 7 , 7]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 128 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 128, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 1, 1]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256*128*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 256, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 128 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 1, 1]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512*512*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 512, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 1, 1]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_b', [36, 2 * self.class_num * (1 + self.n_query)])
            ]
        elif 'resnet10_LAT_out2' == self.backbone:
            config = [
                # # 54*128*28*28
                ('conv2d', [64,128,3,3,2,1]), # out 28 14
                ('bn', [64]),
                ('relu', [True]),

                ('conv2d', [32, 64, 3, 3, 2, 1]), # out 14 7
                ('bn', [32]),
                ('relu', [True]),

                ('conv2d', [16, 32, 3, 3, 2, 1]),  # out 7 4
                ('bn', [16]),
                ('relu', [True]),

                ('conv2d', [8, 16, 3, 3, 2, 1]),  # out 4 2
                ('bn', [8]),
                ('relu', [True]),

                ('conv2d', [2, 8, 3, 3, 1, 1]),  # out 2 2
                ('bn', [2]),
                ('relu', [True]),

                ('avg_pool2d', [2,2,0]),  # out 45*8*1*1

                ('flatten_w',[]),
                # ('linear', [64*3*7*7,512*self.class_num*(self.k_shot+self.n_query)*7*7]),
                ('linear_re', [64 * 3 * 7 * 7, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64 , 3 , 7 , 7]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 128 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 128, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 1, 1]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256*128*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 256, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 128 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 1, 1]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512*512*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 512, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 1, 1]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_b', [36, 2 * self.class_num * (1 + self.n_query)])
            ]
        elif 'resnet10_LAT_out3' == self.backbone:
            config = [
                # 54*256*14*14
                ('conv2d', [128,256,3,3,2,1]), # out 7
                ('bn', [128]),
                ('relu', [True]),

                ('conv2d', [64, 128, 3, 3, 2, 1]), # out 4
                ('bn', [64]),
                ('relu', [True]),

                ('conv2d', [32, 64, 3, 3, 2, 1]),  # out 2
                ('bn', [32]),
                ('relu', [True]),

                ('conv2d', [8, 32, 3, 3, 1, 1]),  # out 2
                ('bn', [8]),
                ('relu', [True]),

                ('conv2d', [2, 8, 3, 3, 1, 1]),  # out 2
                ('bn', [2]),
                ('relu', [True]),

                ('avg_pool2d', [2,2,0]),  # out 45*8*1*1

                ('flatten_w',[]),
                # ('linear', [64*3*7*7,512*self.class_num*(self.k_shot+self.n_query)*7*7]),
                ('linear_re', [64 * 3 * 7 * 7, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64 , 3 , 7 , 7]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 128 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 128, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 1, 1]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256*128*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 256, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 128 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 1, 1]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512*512*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 512, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 1, 1]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_b', [36, 2 * self.class_num * (1 + self.n_query)])
            ]

        elif 'resnet10_LAT_out4' == self.backbone:
            config = [
                # 54*512*7*7
                ('conv2d', [128,512,3,3,2,1]), # out 4
                ('bn', [128]),
                ('relu', [True]),

                ('conv2d', [64, 128, 3, 3, 2, 1]), # out 2
                ('bn', [64]),
                ('relu', [True]),

                ('conv2d', [32, 64, 3, 3, 1, 1]),  # out 2
                ('bn', [32]),
                ('relu', [True]),

                ('conv2d', [8, 32, 3, 3, 1, 1]),  # out 2
                ('bn', [8]),
                ('relu', [True]),

                ('conv2d', [2, 8, 3, 3, 1, 1]),  # out 2
                ('bn', [2]),
                ('relu', [True]),

                ('avg_pool2d', [2,2,0]),  # out 45*8*1*1

                ('flatten_w',[]),
                # ('linear', [64*3*7*7,512*self.class_num*(self.k_shot+self.n_query)*7*7]),
                ('linear_re', [64 * 3 * 7 * 7, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64 , 3 , 7 , 7]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 128 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 128, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 1, 1]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256*128*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 256, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 128 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 1, 1]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512*512*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 512, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 1, 1]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_b', [36, 2 * self.class_num * (1 + self.n_query)])
            ]


        elif 'resnet10_LAT' == self.backbone:
            config = [
                # ('reshape', [-1]),
                ('conv2d', [32,64,3,3,2,1]), # out 28
                ('bn', [32]),
                ('relu', [True]),

                ('conv2d', [16, 32, 3, 3, 2, 1]), # out 14
                ('bn', [16]),
                ('relu', [True]),

                ('conv2d', [8, 16, 3, 3, 2, 1]),  # out 7
                ('bn', [8]),
                ('relu', [True]),

                ('conv2d', [4, 8, 3, 3, 2, 1]),  # out 4
                ('bn', [4]),
                ('relu', [True]),

                ('conv2d', [2, 4, 3, 3, 2, 1]),  # out 2
                ('bn', [2]),
                ('relu', [True]),

                ('avg_pool2d', [2,2,0]),  # out 45*8*1*1

                ('flatten_w',[]),
                # ('linear', [64*3*7*7,512*self.class_num*(self.k_shot+self.n_query)*7*7]),
                ('linear_re', [64 * 3 * 7 * 7, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64 , 3 , 7 , 7]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64 , 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [64 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [64, 64, 3, 3]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [64, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 128 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 128, 3, 3]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [128 * 64 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [128, 64, 1, 1]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [128, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256*128*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 256, 3, 3]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [256 * 128 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [256, 128, 1, 1]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [256, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 3 * 3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512*512*3*3, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 512, 3, 3]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_re', [512 * 256 * 1 * 1, 2 * self.class_num * (1 + self.n_query)]),
                ('reshape_w', [512, 256, 1, 1]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_w', [512, 2 * self.class_num * (1 + self.n_query)]),
                ('linear_b', [36, 2 * self.class_num * (1 + self.n_query)])
            ]

        
        return config, header_config
