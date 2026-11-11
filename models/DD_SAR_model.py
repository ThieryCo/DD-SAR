import torch
from torch.nn import L1Loss, MSELoss

import nets
from .base_model import BaseModel
import loss.loss as loss

class DD_SAR_Model(BaseModel):
    def name(self):
        return 'DD_SAR_Model'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', net='DD_SAR')
        parser.set_defaults(dataset_mode='aligned')

        parser.add_argument('--loss_names', type=str, nargs='+', default=['MIX3'],help='loss names')

        if is_train:
            parser.set_defaults(pool_size=0, no_lsgan=True)

        return parser

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.model_names = ['DD_SAR']
        self.isTrain = opt.isTrain

        self.netDD_SAR = nets.define(opt.init_type, opt.init_gain, self.gpu_ids, opt)
        self.netDD_SAR.to(self.device)

        if self.isTrain:
            self.visual_names = ['denoised', 'clean']

            self.criterion_MIX3 = loss.MIX3Loss()

            self.optimizers = []
            self.optimizer = torch.optim.AdamW(
                self.netDD_SAR.parameters(),
                lr=opt.lr,
                weight_decay=opt.weight_decay,
                betas=(opt.beta1, opt.beta2),
            )
            self.optimizers.append(self.optimizer)

        else:
            self.visual_names = ['denoised']
        self.autocast_ctx = torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=False)

    def set_input(self, input):
        self.noise = input['Noise'].to(self.device, dtype=torch.float32, non_blocking=True)
        self.noise_image_paths = input['Noise_paths']
        if self.isTrain:
            self.clean = input['Clean'].to(self.device, dtype=torch.float32, non_blocking=True)

    def forward(self):
        with self.autocast_ctx:
            self.denoised = self.netDD_SAR(self.noise)
            if self.isTrain:
                self.loss_MIX3 = self.criterion_MIX3(self.denoised, self.clean)
                self.loss = self.loss_MIX3


    def backward(self):
        self.loss.backward()

    def optimize_parameters(self):
        self.forward()
        self.optimizer.zero_grad(set_to_none=True)
        self.backward()
        self.optimizer.step()
