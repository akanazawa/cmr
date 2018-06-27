'''Code adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix'''
import numpy as np
import os
import ntpath
import time
import visdom
from . import visutil as util

class Visualizer():
    def __init__(self, opt):
        # self.opt = opt
        self.display_id = opt.display_id
        self.win_size = opt.display_winsize
        self.name = opt.name
        if self.display_id > 0:
            self.vis = visdom.Visdom(port = opt.display_port)
            self.display_single_pane_ncols = opt.display_single_pane_ncols

        self.log_name = os.path.join(opt.checkpoint_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch):
        if self.display_id > 0: # show images in the browser
            if self.display_single_pane_ncols > 0:
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
    table {border-collapse: separate; border-spacing:4px; white-space:nowrap; text-align:center}
    table td {width: %dpx; height: %dpx; padding: 4px; outline: 4px solid black}
</style>""" % (w, h)
                ncols = self.display_single_pane_ncols
                title = self.name
                label_html = ''
                label_html_row = ''
                nrows = int(np.ceil(len(visuals.items()) / ncols))
                images = []
                idx = 0
                # for label, image_numpy in visuals.items():
                img_keys = visuals.keys()
                list.sort(img_keys)
                for label in img_keys:
                    image_numpy = visuals[label]
                    label_html_row += '<td>%s</td>' % label
                    images.append(image_numpy.transpose([2, 0, 1]))
                    idx += 1
                    if idx % ncols == 0:
                        label_html += '<tr>%s</tr>' % label_html_row
                        label_html_row = ''
                white_image = np.ones_like(image_numpy.transpose([2, 0, 1]))*255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                # pane col = image row
                self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                padding=2, opts=dict(title=title + ' images'))
                label_html = '<table>%s</table>' % label_html
                self.vis.text(table_css + label_html, win = self.display_id + 2,
                              opts=dict(title=title + ' labels'))
            else:
                idx = 1
                for label, image_numpy in visuals.items():
                    self.vis.image(
                        image_numpy.transpose([2,0,1]), opts=dict(title=label),
                        win=self.display_id + idx)
                    idx += 1

    # scalars: dictionary of scalar labels and values
    def plot_current_scalars(self, epoch, counter_ratio, opt, scalars):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(scalars.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([scalars[k] for k in self.plot_data['legend']])
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': self.name + ' loss over time',
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch',
                'ylabel': 'loss'},
            win=self.display_id)

    # scatter plots
    def plot_current_points(self, points, disp_offset=10):
        idx = disp_offset
        for label, pts in points.items():
            #image_numpy = np.flipud(image_numpy)
            self.vis.scatter(
                pts, opts=dict(title=label, markersize=1), win=self.display_id + idx)
            idx += 1

    # scalars: same format as |scalars| of plot_current_scalars
    def print_current_scalars(self, epoch, i, scalars):
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in scalars.items():
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
