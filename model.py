import torch
import torch.nn as nn
import torchvision.models as models
from loss import MMD_loss, CORAL

class TransferModel(nn.Module):
    def __init__(self, base_model: str = 'resnet50', pretrain: bool = True, n_class: int = 31):
        super(TransferModel, self).__init__()
        self.base_model = base_model
        self.pretrain = pretrain
        self.n_class = n_class
        if self.base_model == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            n_features = self.model.fc.in_features
            fc = torch.nn.Linear(n_features, n_class)
            self.model.fc = fc
        else:
            pass
        self.model.fc.weight.data.normal_(0, 0.005)
        self.model.fc.bias.data.fill_(0.1)

    def forward(self, x):
        return self.model(x)
    
    def predict(self, x):
        return self.forward(x)

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
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(TransferNet, self).__init__()
        if base_net == 'resnet50':
            self.base_network = ResNet50Fc()
        else:
            return
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss

        # Comment out the bottleneck layer
        # bottleneck_list = [nn.Linear(self.base_network.output_num(), bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        # self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(self.base_network.output_num(), width), nn.ReLU(), nn.Dropout(0.5), nn.Linear(width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        # Comment out the bottleneck layer initialization
        # self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        # self.bottleneck_layer[0].bias.data.fill_(0.1)
        for i in range(2):
            self.classifier_layer[i * 3].weight.data.normal_(0, 0.01)
            self.classifier_layer[i * 3].bias.data.fill_(0.0)

        # Gaussian parameters for KL divergence
        self.fc_mu_input = nn.Linear(self.base_network.output_num(), bottleneck_width)
        self.fc_log_var_input = nn.Linear(self.base_network.output_num(), bottleneck_width)
        # Comment out the bottleneck Gaussian parameters
        # self.fc_mu_bottleneck = nn.Linear(bottleneck_width, bottleneck_width)
        # self.fc_log_var_bottleneck = nn.Linear(bottleneck_width, bottleneck_width)

    def forward(self, source, target=None, return_fc_features=False, return_gaussian_params=False):
        source_features = self.base_network(source)
        if target is not None:
            target_features = self.base_network(target)
        
        # Comment out the bottleneck layer usage
        # if self.use_bottleneck:
        #     source_bottleneck = self.bottleneck_layer(source_features)
        #     if target is not None:
        #         target_bottleneck = self.bottleneck_layer(target_features)
        
        transfer_loss = self.adapt_loss(source_features, target_features, self.transfer_loss) if target is not None else None

        source_clf = self.classifier_layer(source_features)

        mu_input = self.fc_mu_input(source_features)
        log_var_input = self.fc_log_var_input(source_features)

        # Comment out the bottleneck Gaussian parameters usage
        # mu_bottleneck = self.fc_mu_bottleneck(source_bottleneck)
        # log_var_bottleneck = self.fc_log_var_bottleneck(source_bottleneck)

        if return_fc_features:
            return (source_features, source_clf, target_features)
        if return_gaussian_params:
            # Comment out the bottleneck Gaussian parameters return
            # return (source_clf, transfer_loss, mu_input, log_var_input, mu_bottleneck, log_var_bottleneck)
            return (source_clf, transfer_loss, mu_input, log_var_input)
        return source_clf, transfer_loss

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            # Your own loss
            loss = 0
        return loss

    def predict(self, x):
        features = self.base_network(x)
        clf = self.classifier_layer(features)
        return clf
