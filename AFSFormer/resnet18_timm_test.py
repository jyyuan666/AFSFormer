import torch
import timm

model = timm.create_model('swsl_resnet18', features_only=True, pretrained=True, output_stride=32, out_indices=(1, 2, 3, 4))
net = timm.create_model('swsl_resnet18', features_only=True, pretrained=True, output_stride=32)

encoder_channels_0 = net.feature_info.channels()
print('net {} channles:'.format(encoder_channels_0))
encoder_channels = model.feature_info.channels()
print(encoder_channels)
input = torch.randn(1,3,1024,1024)

all_features = model(input)
print('All {} Features: '.format(len(all_features)))
for i in range(len(all_features)):
    print('feature {} shape: {}'.format(i, all_features[i].shape))

all_features_0 = net(input)
print('All {} Features: '.format(len(all_features_0)))
for i in range(len(all_features_0)):
    print('feature {} shape: {}'.format(i, all_features_0[i].shape))


res1,res2,res3,res4 = model(input)
print(res1.shape,res2.shape,res3.shape,res4.shape)
