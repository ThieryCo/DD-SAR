import os
from options.test_options import TestOptions
from data import create_data_loader
from models import create_model
from util.visualizer import save_images
from util import my_html


if __name__ == '__main__':
    opt = TestOptions().parse()

    data_loader = create_data_loader(opt)
    dataset = data_loader.load_data()
    model = create_model(opt)
    model.setup(opt)

    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))
    webpage = my_html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    model.eval()
    print("Evaluating...")
    for i, data in enumerate(dataset):
        if i >= opt.num_test:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()
        if i % 10 == 0:
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
