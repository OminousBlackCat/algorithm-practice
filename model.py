import encoder
import dataset
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


class Model(nn.Module):
    def __init__(self, input_channel=3, medium_channel=192):
        super(Model, self).__init__()
        self.input_channel = input_channel
        self.medium_channel = medium_channel
        self.encoder = encoder.Encoder(input_channel, medium_channel, medium_channel)
        self.decoder = encoder.Decoder(medium_channel, medium_channel)

    def forward(self, input_x):
        feature = self.encoder.forward(input_x)
        quant_noise_feature = torch.zeros(feature.shape).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        comressed_feature = feature + quant_noise_feature
        out_z = self.decoder.forward(comressed_feature)
        mes_loss = torch.mean((out_z[:, :, : 1000, : 1000] - input_x).pow(2))
        return out_z, mes_loss


def saveModel(my_model: Model):
    torch.save(my_model.state_dict(), "./myModel.pth")


def validate(my_model: Model, my_testSet: DataLoader, validation_cnt, device):
    my_model.eval()
    with torch.no_grad():
        for img in my_testSet:
            inputs = img.to(device)
            outputs, _ = my_model(inputs)
            plt.imsave("data/save/validate" + str(validation_cnt) + ".png", outputs.cpu()[0][0].detach().numpy())
            validation_cnt += 1
    return validation_cnt


def train(num_epochs, my_model: Model, my_dataset: DataLoader, my_testSet: DataLoader, my_opt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("将使用", device)
    my_model.to(device)
    validation_cnt = 0
    scheduler = torch.optim.lr_scheduler.StepLR(my_opt, step_size=8, gamma=0.4)
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, img in enumerate(my_dataset, 0):
            inputs = img.to(device)
            my_opt.zero_grad()
            outputs, mes_loss = my_model(inputs)
            mes_loss.backward()
            my_opt.step()
            running_loss += mes_loss.item()
            if i % 100 == 99:
                print('[%d, %5d] loss: %.8f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

        def clip_gradient(c_optimizer, grad_clip):
            for group in c_optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)

        clip_gradient(optimizer, 5)
        scheduler.step()
        print("epoch" + str(epoch + 1) + ": Done!")
        validation_cnt = validate(model, my_testSet, validation_cnt, device)


if __name__ == '__main__':
    # 定义模型
    model = Model(1)
    # 获得数据集
    trainDataset = dataset.SunData("data/train", dataset.load_image_list("data/train"))
    testDataset = dataset.SunData("data/validation", dataset.load_image_list("data/validation"))
    trainDataLoader = DataLoader(trainDataset, batch_size=2)
    testDataLoader = DataLoader(testDataset)
    # 一些超参数
    epochs = 10
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    train(100, model, trainDataLoader, testDataLoader, optimizer)
    print("fin")
