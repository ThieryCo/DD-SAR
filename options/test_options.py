from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--data_root_noise', type=str, default='./datasets/real/SAR_intensity', help='path to noise (test) images')

        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')

        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results',
                            help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')

        parser.add_argument('--num_test', type=int, default=50000, help='how many test images to run')

        parser.set_defaults(num_threads=1)
        parser.set_defaults(serial_batches = True)
        parser.set_defaults(display_id = -1)
        self.isTrain = False
        return parser
