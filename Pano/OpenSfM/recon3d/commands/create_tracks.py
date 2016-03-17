import logging

from recon3d import dataset
from recon3d import matching

logger = logging.getLogger(__name__)


class Command:
    name = 'create_tracks'
    help = "Link matches pair-wise matches into tracks"

    def add_arguments(self, parser):
        parser.add_argument('dataset', help='dataset to process')

    def run(self, args):
        data = dataset.DataSet(args.dataset)
        images = data.images()

        # Read local features
        logging.info('reading features')
        features = {}
        colors = {}
        for im in images:
            p, f, c = data.load_features(im)
            features[im] = p[:, :2]
            colors[im] = c
        # print 'features: ',features
        # raw_input()
        # print 'colors: ',colors
        # raw_input()
        # Read matches
        matches = {}
        # print 'images: ',images
        # raw_input()
        for im1 in images:
            try:
                # print 'im1: ',im1
                # raw_input()
                im1_matches = data.load_matches(im1)
                # print 'im1_matches: ',im1_matches.keys()
                # raw_input()
            except IOError:
                # print 'im1 err: ',im1
                # raw_input()
                continue
            for im2 in im1_matches:
                matches[im1, im2] = im1_matches[im2]
                # print 'im1_matches[im2]: ',im1_matches[im2]
                # raw_input()
        tracks_graph = matching.create_tracks_graph(features, colors, matches,
                                                    data.config)
        data.save_tracks_graph(tracks_graph)
