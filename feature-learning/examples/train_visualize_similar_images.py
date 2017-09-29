
from feature_learning.similar_image_feature_learner import SimilarImageFeatureLearner
from feature_learning.utils.integer_searcher import IntegerSearcher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-images_directory',
    required=True,
    help='Path to directory with images'
)
args = parser.parse_args()

images_path = args.images_directory

learner = SimilarImageFeatureLearner(images_path)

image_file_paths, feature_vectors = learner.train()

visualizer = IntegerSearcher(
    image_paths=image_file_paths,
    image_feature_vectors=feature_vectors
)

visualizer.visualize_similar_images()
