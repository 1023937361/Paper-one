from VGG11 import train as VGG11
from VGG11 import predict as VGG11_pre
from VGG13 import train as VGG13
from VGG13 import predict as VGG13_pre
from VGG16 import train as VGG16
from VGG16 import predict as VGG16_pre
from VGG19 import train as VGG19
from VGG19 import predict as VGG19_pre

from ResNet18 import train as ResNet18
from ResNet18 import predict as ResNet18_pre
from ResNet34 import train as ResNet34
from ResNet34 import predict as ResNet34_pre
from ResNet50 import train as ResNet50
from ResNet50 import predict as ResNet50_pre
from ResNet101 import train as ResNet101
from ResNet101 import predict as ResNet101_pre

from Densenet121 import train as Densenet121
from Densenet121 import predict as Densenet121_pre
from Densenet169 import train as Densenet169
from Densenet169 import predict as Densenet169_pre

from Inception_V3 import train as Inceptionv3
from Inception_V3 import  predict as Inceptionv3_pre

from Xception import train as Xception
from Xception import  predict as Xception_pre

from Alexnet import train as alex
from Alexnet import predict as alex_pre

from Googlenet import train as googlenet
from Googlenet import  predict as googlenet_pre

from Mobilenet_V2 import train as mobilenet
from Mobilenet_V2 import predict as mobilenet_pre

from Shufflenet_V2x10 import train as shufflenetx10
from Shufflenet_V2x10 import  predict as shufflenetx10_pre
from Shufflenet_V2x05 import train as shufflenetx05
from Shufflenet_V2x05 import  predict as shufflenetx05_pre

from InceptionResnetV1 import train as InceptionResnetV1
from InceptionResnetV1 import predict as InceptionResnetV1_pre

from vit_5 import train as vit
from vit_5 import  predict as vit_pre

from Botnet import  train as botnet
from  Botnet import  predict as botnet_pre

from Deit import  train as deit
from Deit import  predict as deit_pre

from T2T_VIT import train as t2t
from T2T_VIT import  predict as t2t_pre




from VGG11_Padding import train as VGG11_pad
from VGG11_Padding import predict as VGG11_pre_pad
from VGG13_Padding import train as VGG13_pad
from VGG13_Padding import predict as VGG13_pre_pad
from VGG16_Padding import train as VGG16_pad
from VGG16_Padding import predict as VGG16_pre_pad
from VGG19_Padding import train as VGG19_pad
from VGG19_Padding import predict as VGG19_pre_pad

from ResNet18_Padding import train as ResNet18_pad
from ResNet18_Padding import predict as ResNet18_pre_pad
from ResNet34_Padding import train as ResNet34_pad
from ResNet34_Padding import predict as ResNet34_pre_pad
from ResNet50_Padding import train as ResNet50_pad
from ResNet50_Padding import predict as ResNet50_pre_pad
from ResNet101_Padding import train as ResNet101_pad
from ResNet101_Padding import predict as ResNet101_pre_pad

from Densenet121_Padding import train as Densenet121_pad
from Densenet121_Padding import predict as Densenet121_pre_pad
from Densenet169_Padding import train as Densenet169_pad
from Densenet169_Padding import predict as Densenet169_pre_pad

from Inception_V3_Padding import train as Inceptionv3_pad
from Inception_V3_Padding import  predict as Inceptionv3_pre_pad

from Xception_Padding import train as Xception_pad
from Xception_Padding import  predict as Xception_pre_pad

from Alexnet_Padding import train as alex_pad
from Alexnet_Padding import predict as alex_pre_pad

from Googlenet_Padding import train as googlenet_pad
from Googlenet_Padding import  predict as googlenet_pre_pad

from Mobilenet_V2_Padding import train as mobilenet_pad
from Mobilenet_V2_Padding import predict as mobilenet_pre_pad

from Shufflenet_V2x10_Padding import train as shufflenetx10_pad
from Shufflenet_V2x10_Padding import  predict as shufflenetx10_pre_pad
from Shufflenet_V2x05_Padding import train as shufflenetx05_pad
from Shufflenet_V2x05_Padding import  predict as shufflenetx05_pre_pad

from InceptionResnetV1_padding import train as InceptionResnetV1_pad
from InceptionResnetV1_padding import predict as InceptionResnetV1_pre_pad

from vit_5_padding import train as vit_pad
from vit_5_padding import  predict as vit_pre_pad

from Botnet_padding import  train as botnet_pad
from  Botnet_padding import  predict as botnet_pre_pad

from Deit_padding import  train as deit_pad
from Deit_padding import  predict as deit_pre_pad

from T2T_VIT_padding import train as t2t_pad
from T2T_VIT_padding import  predict as t2t_pre_pad

if __name__ == '__main__':
    VGG11.main()
    VGG11_pre.main()
    VGG13.main()
    VGG13_pre.main()
    VGG16.main()
    VGG16_pre.main()
    VGG19.main()
    VGG19_pre.main()

    ResNet18.main()
    ResNet18_pre.main()
    ResNet34.main()
    ResNet34_pre.main()
    ResNet50.main()
    ResNet50_pre.main()
    ResNet101.main()
    ResNet101_pre.main()

    Densenet121.main()
    Densenet121_pre.main()
    Densenet169.main()
    Densenet169_pre.main()

    Inceptionv3.main()
    Inceptionv3_pre.main()

    Xception.main()
    Xception_pre.main()

    alex.main()
    alex_pre.main()

    googlenet.main()
    googlenet_pre.main()

    mobilenet.main()
    mobilenet_pre.main()

    shufflenetx10.main()
    shufflenetx10_pre.main()
    shufflenetx05.main()
    shufflenetx05_pre.main()

    InceptionResnetV1.main()
    InceptionResnetV1_pre.main()

    vit.main()
    vit_pre.main()

    botnet.main()
    botnet_pre.main()

    deit.main()
    deit_pre.main()

    t2t.main()
    t2t_pre.main()




    VGG11_pad.main()
    VGG11_pre_pad.main()
    VGG13_pad.main()
    VGG13_pre_pad.main()
    VGG16_pad.main()
    VGG16_pre_pad.main()
    VGG19_pad.main()
    VGG19_pre_pad.main()

    ResNet18_pad.main()
    ResNet18_pre_pad.main()
    ResNet34_pad.main()
    ResNet34_pre_pad.main()
    ResNet50_pad.main()
    ResNet50_pre_pad.main()
    ResNet101_pad.main()
    ResNet101_pre_pad.main()

    Densenet121_pad.main()
    Densenet121_pre_pad.main()
    Densenet169_pad.main()
    Densenet169_pre_pad.main()

    Inceptionv3_pad.main()
    Inceptionv3_pre_pad.main()

    Xception_pad.main()
    Xception_pre_pad.main()

    alex_pad.main()
    alex_pre_pad.main()

    googlenet_pad.main()
    googlenet_pre_pad.main()

    mobilenet_pad.main()
    mobilenet_pre_pad.main()

    shufflenetx10_pad.main()
    shufflenetx10_pre_pad.main()
    shufflenetx05_pad.main()
    shufflenetx05_pre_pad.main()

    InceptionResnetV1_pad.main()
    InceptionResnetV1_pre_pad.main()

    vit_pad.main()
    vit_pre_pad.main()

    botnet_pad.main()
    botnet_pre_pad.main()

    deit_pad.main()
    deit_pre_pad.main()

    t2t_pad.main()
    t2t_pre_pad.main()
