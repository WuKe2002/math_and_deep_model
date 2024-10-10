
from archs.lednet_arch import LEDNet

import cv2
import torch
import torch.nn as nn

model = LEDNet().to('cuda')
# 加载模型权重
model.load_state_dict(torch.load('./weights/lednet.pth')["params"])

# 读取images文件夹下的图像数据
img_path = './images/0000_0082.png'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.transpose(2, 0, 1)
# 转化为tensor输入到模型中
img = torch.from_numpy(img).float().unsqueeze(0)
img = img / 255.
img = img.cuda()

out = model(img)

# 可视化输出
out = out.cpu().detach().numpy()
out = out.transpose(0, 2, 3, 1)
out = out[0]
out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
cv2.imshow('out', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('out.png', out)

# 打印模型
print(model)
#print(model._modules.items())


features_in_hook, features_out_hook = [], []
def hook(module, fea_in, fea_out):
    features_in_hook.append(fea_in)
    features_out_hook.append(fea_out)
    return None

layer_name = 'PPM1'

for (name, module) in model.named_modules():
    if name == layer_name:
        module.register_forward_hook(hook=hook)

output = model(img)

print(output.shape)
print(out.shape)
print(features_in_hook)
print(features_out_hook)

#可视化feature_in_hook,features_out_hook
'''
for i in range(len(features_in_hook)):
    out = features_in_hook[i][0].transpose(0, 2, 3, 1)
    out = out[0]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imshow('features_in_hook'+str(i), out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('features_in_hook'+str(i)+'.png', out)
    out = features_out_hook[i][0].transpose(0, 2, 3, 1)
    out = out[0]
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    cv2.imshow('features_out_hook'+str(i), out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('features_out_hook'+str(i)+'.png', features_out_hook[i][0].cpu().detach().numpy())
    '''

# 使用torch.linalg.matrix_rank计算矩阵的特征值
print(torch.linalg.matrix_rank(features_in_hook[0][0]))
print(torch.linalg.matrix_rank(features_out_hook[0][0]))
