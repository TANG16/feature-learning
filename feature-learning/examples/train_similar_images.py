
from feature_learning.feature_learners.similar_image_feature_learner import SimilarImageFeatureLearner
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
