import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50Fc(nn.Module):
    def __init__(self):
        super(ResNet50Fc, self).__init__()
        model_resnet50 = models.resnet50(pretrained=True)
        self.conv1 = model_resnet50.conv1
        self.bn1 = model_resnet50.bn1
        self.relu = model_resnet50.relu
        self.maxpool = model_resnet50.maxpool
        self.layer1 = model_resnet50.layer1
        self.layer2 = model_resnet50.layer2
        self.layer3 = model_resnet50.layer3
        self.layer4 = model_resnet50.layer4
        self.avgpool = model_resnet50.avgpool
        self.__in_features = model_resnet50.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class TransferNet(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256):
        super(TransferNet, self).__init__()
        if base_net == 'resnet50':
            self.base_network = ResNet50Fc()
        else:
            raise NotImplementedError("Only ResNet50 is supported currently.")
        
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        
        bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        
        self.fc = nn.Linear(bottleneck_width, num_class)
        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)

    def forward(self, source, target=None, return_gaussian_params=False):
        source_features = self.base_network(source)
        if target is not None:
            target_features = self.base_network(target)
        if self.use_bottleneck:
            source_bottleneck = self.bottleneck_layer(source_features)
            if target is not None:
                target_bottleneck = self.bottleneck_layer(target_features)
        transfer_loss = None
        if target is not None:
            if self.transfer_loss == 'mmd':
                transfer_loss = MMD_loss()(source_bottleneck, target_bottleneck)
            elif self.transfer_loss == 'coral':
                transfer_loss = CORAL(source_bottleneck, target_bottleneck)
        classifier_output = self.fc(source_bottleneck)
        if return_gaussian_params:
            mu = self.fc_mu(source_bottleneck)
            log_var = self.fc_log_var(source_bottleneck)
            return classifier_output, transfer_loss, mu, log_var
        return classifier_output, transfer_loss
