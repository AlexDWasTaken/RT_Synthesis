import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d, ResBlock3d, SPADEResnetBlock
from modules.dense_motion import DenseMotionNetwork
from timeit import default_timer
from torch.nn.utils import prune

class OcclusionAwareGenerator(nn.Module):
    """
    Generator follows NVIDIA architecture.
    """

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.resblocks_2d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_2d.add_module('2dr' + str(i), ResBlock2d(out_features, kernel_size=3, padding=1))

        up_blocks = []
        for i in range(num_down_blocks):
            in_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = max(block_expansion, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.up_blocks = nn.ModuleList(up_blocks)

        self.final = nn.Conv2d(block_expansion, image_channel, kernel_size=(7, 7), padding=(3, 3))
        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        tmp = default_timer()
        
        out = self.first(source_image)
        
        time1 = default_timer()
        print("first step:",time1-tmp)
        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        out = self.second(out)
        bs, c, h, w = out.shape
        # print(out.shape)
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

            # output_dict["deformed"] = self.deform_input(source_image, deformation)  # 3d deformation cannot deform 2d image

        # Decoding part
        out = self.resblocks_2d(out)
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = torch.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict


class SPADEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        ic = 256
        oc = 64
        norm_G = 'spadespectralinstance'
        label_nc = 256
        
        self.fc = nn.Conv2d(ic, 2 * ic, 3, padding=1)
        self.G_middle_0 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_1 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_2 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_3 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_4 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.G_middle_5 = SPADEResnetBlock(2 * ic, 2 * ic, norm_G, label_nc)
        self.up_0 = SPADEResnetBlock(2 * ic, ic, norm_G, label_nc)
        self.up_1 = SPADEResnetBlock(ic, oc, norm_G, label_nc)
        self.conv_img = nn.Conv2d(oc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)
        self.up_simple = SPADEResnetBlock(2 * ic, oc, norm_G, label_nc)
        
    def forward(self, feature):
        print("@@@@@@@@@@---inside-decoder---@@@@@@@@@@@@@@@")
        torch.cuda.synchronize()
        time0 = default_timer()
        
        seg = feature
        x = self.fc(feature)
        #x = self.G_middle_0(x, seg)
        
        torch.cuda.synchronize()
        time1 = default_timer()
        print("fully connected:", time1-time0)
        
        #x = self.G_middle_1(x, seg)
        #x = self.G_middle_2(x, seg) 
        x = self.G_middle_3(x, seg)
        x = self.G_middle_4(x, seg)
        x = self.G_middle_5(x, seg)
        
        torch.cuda.synchronize()
        time2 = default_timer()
        print("G_Middles:", time2-time1) #0.049
        
        """x = self.up(x)                
        x = self.up_0(x, seg)         # 256, 128, 128
        x = self.up(x)                
        x = self.up_1(x, seg)         # 64, 256, 256"""
        x = self.up(x)
        x = self.up_simple(x, seg)
        x = self.up(x)

        torch.cuda.synchronize()
        time3 = default_timer()
        print("ups:", time3-time2)
        
        x = self.conv_img(F.leaky_relu(x, 2e-1)) #0.07
        # x = torch.tanh(x)
        x = torch.sigmoid(x)
        
        torch.cuda.synchronize()
        time4 = default_timer()
        print("conv_img:", time4-time3)
        
        print("@@@@@@@@@@@@@@@@@@@@---decoder-end---@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        
        return x
    
    def prune_conv_img(self):
        # prune conv_img layer using l1unstructured
        prune.l1_unstructured(self.conv_img, name='weight', amount=0.5)
        prune.remove(self.conv_img, 'weight')
        
        


class OcclusionAwareSPADEGenerator(nn.Module):

    def __init__(self, image_channel, feature_channel, num_kp, block_expansion, max_features, num_down_blocks, reshape_channel, reshape_depth,
                 num_resblocks, estimate_occlusion_map=False, dense_motion_params=None, estimate_jacobian=False):
        super(OcclusionAwareSPADEGenerator, self).__init__()

        if dense_motion_params is not None:
            self.dense_motion_network = DenseMotionNetwork(num_kp=num_kp, feature_channel=feature_channel,
                                                           estimate_occlusion_map=estimate_occlusion_map,
                                                           **dense_motion_params)
        else:
            self.dense_motion_network = None

        self.first = SameBlock2d(image_channel, block_expansion, kernel_size=(3, 3), padding=(1, 1))

        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        self.down_blocks = nn.ModuleList(down_blocks)

        self.second = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape_channel = reshape_channel
        self.reshape_depth = reshape_depth

        self.resblocks_3d = torch.nn.Sequential()
        for i in range(num_resblocks):
            self.resblocks_3d.add_module('3dr' + str(i), ResBlock3d(reshape_channel, kernel_size=3, padding=1))

        out_features = block_expansion * (2 ** (num_down_blocks))
        self.third = SameBlock2d(max_features, out_features, kernel_size=(3, 3), padding=(1, 1), lrelu=True)
        self.fourth = nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=1, stride=1)

        self.estimate_occlusion_map = estimate_occlusion_map
        self.image_channel = image_channel

        self.decoder = SPADEDecoder()
        #self.decoder.prune_conv_img()

    def deform_input(self, inp, deformation):
        _, d_old, h_old, w_old, _ = deformation.shape
        _, _, d, h, w = inp.shape
        if d_old != d or h_old != h or w_old != w:
            deformation = deformation.permute(0, 4, 1, 2, 3)
            deformation = F.interpolate(deformation, size=(d, h, w), mode='trilinear')
            deformation = deformation.permute(0, 2, 3, 4, 1)
        return F.grid_sample(inp, deformation)

    def forward(self, source_image, kp_driving, kp_source):
        # Encoding (downsampling) part
        print("#############################################")
        tmp = default_timer()
        
        out = self.first(source_image)
        
        #print(out)
        torch.cuda.synchronize()
        
        time1 = default_timer()
        
        print("self.first:",time1-tmp)
        
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)
        
        torch.cuda.synchronize()
        time2 = default_timer()
        
        print("self.downblocks:", time2-time1)
            
        out = self.second(out)
        bs, c, h, w = out.shape
        feature_3d = out.view(bs, self.reshape_channel, self.reshape_depth, h ,w) 
        feature_3d = self.resblocks_3d(feature_3d)

        torch.cuda.synchronize()
        time3 = default_timer()
        print("second+feature3d:", time3-time2)
        
        
        # Transforming feature representation according to deformation and occlusion
        output_dict = {}
        if self.dense_motion_network is not None:
            dense_motion = self.dense_motion_network(feature=feature_3d, kp_driving=kp_driving,
                                                     kp_source=kp_source)
            output_dict['mask'] = dense_motion['mask']

            if 'occlusion_map' in dense_motion:
                occlusion_map = dense_motion['occlusion_map']
                output_dict['occlusion_map'] = occlusion_map
            else:
                occlusion_map = None
            deformation = dense_motion['deformation']
            out = self.deform_input(feature_3d, deformation)

            bs, c, d, h, w = out.shape
            out = out.view(bs, c*d, h, w)
            out = self.third(out)
            out = self.fourth(out)

            if occlusion_map is not None:
                if out.shape[2] != occlusion_map.shape[2] or out.shape[3] != occlusion_map.shape[3]:
                    occlusion_map = F.interpolate(occlusion_map, size=out.shape[2:], mode='bilinear')
                out = out * occlusion_map

        torch.cuda.synchronize()
        time4 = default_timer()
        print("Transform feature representation:", time4-time3)
        
        
        # Decoding part
        out = self.decoder(out)

        output_dict["prediction"] = out

        torch.cuda.synchronize()
        print("decode:", default_timer() - time4)
        print()
               
        return output_dict