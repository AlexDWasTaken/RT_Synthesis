import torch
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--original_checkpoint', type=str, default='model.pth')
    args.add_argument('--new_checkpoint', type=str, default='model.pth')
    opt = args.parse_args()
    
    old_checkpoint = torch.load(opt.original_checkpoint)
    new_checkpoint = torch.load(opt.new_checkpoint)
    
    new_checkpoint_upd = new_checkpoint.copy()
    new_checkpoint_upd['kp_detector'] = old_checkpoint['kp_detector']
    new_checkpoint_upd['he_estimator'] = old_checkpoint['he_estimator']
    
    torch.save(new_checkpoint_upd, opt.new_checkpoint + 'upd.tar')
        
    #dict_keys(['generator', 'discriminator', 'kp_detector', 'he_estimator', 'optimizer_generator', 'optimizer_discriminator', 'optimizer_kp_detector', 'optimizer_he_estimator', 'epoch'])