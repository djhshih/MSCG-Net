from model import *
from config.configs_kf import *

# score 0.547, no TTA
ckpt1 = {
    'net': 'MSCG-Rx50',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    #'snapshot': '../ckpt/epoch_8_loss_0.99527_acc_0.82278_acc-cls_0.60967_'
    #            'mean-iu_0.48098_fwavacc_0.70248_f1_0.62839_lr_0.0000829109.pth'
    #'snapshot': '../ckpt/epoch_17_loss_0.93258_acc_0.83161_acc-cls_0.66267_mean-iu_0.51705_fwavacc_0.72544_f1_0.65833_lr_0.0000707703.pth'
    'snapshot': '../ckpt/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-reproduce_ACW_loss2_adax/epoch_24_loss_0.91975_acc_0.83162_acc-cls_0.66588_mean-iu_0.51931_fwavacc_0.72395_f1_0.66157_lr_0.0000570898.pth'
}

ckpt1u = {
    'net': 'MSCG-Rx50',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_62_loss_1.24670_acc_0.74541_acc-cls_0.41243_mean-iu_0.31532_fwavacc_0.59888_f1_0.43747_lr_0.0000014140.pth'
}

ckpt1uc = {
    'net': 'MSCG-Rx50',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    #'snapshot': '../ckpt/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-untrained_ce-loss/epoch_6_loss_1.02698_acc_0.72439_acc-cls_0.33812_mean-iu_0.21389_fwavacc_0.63850_f1_0.30633_lr_0.0000845121.pth'
    'snapshot': '../ckpt/MSCG-Rx50/Agriculture_NIR-RGB_kf-0-0-untrained_ce-loss/epoch_32_loss_0.76079_acc_0.73614_acc-cls_0.39399_mean-iu_0.23909_fwavacc_0.63335_f1_0.34224_lr_0.0000394267.pth'
}

# score 0.550 , no TTA
ckpt2 = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    #'snapshot': '../ckpt/epoch_15_loss_1.03019_acc_0.83952_acc-cls_0.70245_'
    #            'mean-iu_0.54833_fwavacc_0.73482_f1_0.69034_lr_0.0001076031.pth'
    #'snapshot': '../ckpt/epoch_13_loss_1.03088_acc_0.83075_acc-cls_0.66848_mean-iu_0.53545_fwavacc_0.71870_f1_0.67404_lr_0.0001119694.pth'
    'snapshot': '../ckpt/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-reproduce/epoch_19_loss_1.02266_acc_0.83416_acc-cls_0.67662_mean-iu_0.54160_fwavacc_0.72626_f1_0.67785_lr_0.0000974746.pth'
}

ckpt2u = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_44_loss_0.73879_acc_0.70490_acc-cls_0.30367_mean-iu_0.10680_fwavacc_0.66691_f1_0.45351_lr_0.0000218068.pth'
}

ckpt2uc = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/MSCG-Rx101/Agriculture_NIR-RGB_kf-0-0-untrained_ce-loss/epoch_31_loss_1.72881_acc_0.67471_acc-cls_0.28925_mean-iu_0.18900_fwavacc_0.55344_f1_0.27563_lr_0.0000602585.pth'
}

ckpt3 = {
    'net': 'MSCG-Rx101',
    'data': 'Agriculture',
    'bands': ['NIR','RGB'],
    'nodes': (32,32),
    'snapshot': '../ckpt/epoch_15_loss_0.88412_acc_0.88690_acc-cls_0.78581_'
                'mean-iu_0.68205_fwavacc_0.80197_f1_0.80401_lr_0.0001075701.pth'
}

# ckpt1 + ckpt2, test score 0.599,
# ckpt1 + ckpt2 + ckpt3, test score 0.608


def get_net(ckpt=ckpt1):
    net = load_model(name=ckpt['net'],
                     classes=7,
                     node_size=ckpt['nodes'])

    net.load_state_dict(torch.load(ckpt['snapshot']))
    net.cuda()
    net.eval()
    return net


def loadtestimg(test_files):

    id_dict = test_files[IDS]
    image_files = test_files[IMG]
    # mask_files = test_files[GT]

    for key in id_dict.keys():
        for id in id_dict[key]:
            if len(image_files) > 1:
                imgs = []
                for i in range(len(image_files)):
                    filename = image_files[i].format(id)
                    path, _ = os.path.split(filename)
                    if path[-3:] == 'nir':
                        # img = imload(filename, gray=True)
                        img = np.asarray(Image.open(filename), dtype='uint8')
                        img = np.expand_dims(img, 2)

                        imgs.append(img)
                    else:
                        img = imload(filename)
                        imgs.append(img)
                image = np.concatenate(imgs, 2)
            else:
                filename = image_files[0].format(id)
                path, _ = os.path.split(filename)
                if path[-3:] == 'nir':
                    # image = imload(filename, gray=True)
                    image = np.asarray(Image.open(filename), dtype='uint8')
                    image = np.expand_dims(image, 2)
                else:
                    image = imload(filename)
            # label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')

            yield image


def loadids(test_files):
    id_dict = test_files[IDS]

    for key in id_dict.keys():
        for id in id_dict[key]:
            yield id


def loadgt(test_files):
    id_dict = test_files[IDS]
    mask_files = test_files[GT]
    for key in id_dict.keys():
        for id in id_dict[key]:
            label = np.asarray(Image.open(mask_files.format(id)), dtype='uint8')
            yield label
