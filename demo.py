import os
import string
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from utils import TokenLabelConverter
from models import Model
from utils import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def run_model_acds_str(image_tensors, model, converter, opt,name):
    image = image_tensors.to(device)
    batch_size = image.shape[0]
    attens, char_preds,attention_mask= model(image, is_eval=True) # final

    # char pred
    _, char_pred_index = char_preds.topk(1, dim=-1, largest=True, sorted=True)
    char_pred_index = char_pred_index.view(-1, converter.batch_max_length)
    length_for_pred = torch.IntTensor([converter.batch_max_length - 1] * batch_size).to(device)
    char_preds_str = converter.char_decode(char_pred_index[:, 1:], length_for_pred)
    
    index = 0

    # char
    char_pred = char_preds_str[index]
    char_pred_EOS = char_pred.find('[s]')
    char_pred = char_pred[:char_pred_EOS]  # prune after "end of sentence" token ([s])

    return char_pred

def load_img(img_path, opt):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((opt.imgW, opt.imgH), Image.BICUBIC)
    img_arr = np.array(img)
    img_tensor = transforms.ToTensor()(img)
    image_tensor = img_tensor.unsqueeze(0)
    return image_tensor
    

def test(opt):
    """ model configuration """
    converter = TokenLabelConverter(opt)
    opt.num_class = len(converter.character)
    
    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model).to(device)
    
    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device))


    # load img
    if os.path.isdir(opt.demo_imgs):
        imgs = [os.path.join(opt.demo_imgs, fname) for fname in os.listdir(opt.demo_imgs)]
        imgs = [img for img in imgs if img.endswith('.jpg') or img.endswith('.png')]
    else:
        imgs = [opt.demo_imgs]
        
    for img in imgs:
        opt.demo_imgs = img
        img_tensor = load_img(opt.demo_imgs, opt)

        """ evaluation """
        model.eval()
        opt.eval = True
        with torch.no_grad():
            char_pred=run_model_acds_str(img_tensor, model, converter, opt,os.path.basename(img))



if __name__ == '__main__':
    opt = get_args(is_train=False)

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    
    opt.saved_model = opt.model_dir
    test(opt)
