import argparse
import numpy as np
import matplotlib.colors
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Generate simple "art" from a Markov chain')
parser.add_argument('--size', type=int, default=256)
parser.add_argument('--colors', nargs='+', default=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'],
                    help='which colours to use, available: [\'b\', \'g\', \'r\', \'c\', \'m\', \'y\', \'k\', \'w\']')
parser.add_argument('--bias_min', type=int, default=10,
                    help='minimum amount to bias a transition')
parser.add_argument('--bias_max', type=int, default=25,
                    help='maximum amount to bias a transition')
parser.add_argument('--n_biased', type=int, default=4,
                    help='how many transitions to bias')
parser.add_argument('--n_images', type=int, default=10,
                    help='how many images to generate')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed')
args = parser.parse_args()

if args.seed is not None:
    np.random.seed(args.seed)

#making sure colours are correct

available_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

for color in args.colors:
    if color not in available_colors:
        raise ValueError(f'Colors must be one of {available_colors}, you gave: {color}')

#generating transition probabilities

for n in range(args.n_images):

    transition_probs = np.ones((len(args.colors),len(args.colors)))

    for i in transition_probs:
        for _ in range(args.n_biased):
            idx = np.random.randint(len(i))
            i[idx] = np.random.randint(args.bias_min, args.bias_max)

    #normalize transition probabilities

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    normalized_transition_probs = softmax(transition_probs)

    #generate the "coded" image, each pixel is an integer representing a colour

    def colour_lookup(colours, code):
        colour = colours[int(code)]
        return np.array(matplotlib.colors.to_rgb(colour))

    coded_image = np.zeros(args.size**2)

    start = np.random.randint(len(args.colors))

    for i, _ in enumerate(coded_image):
        
        transition = np.random.choice(len(args.colors), p=normalized_transition_probs[:,start])
        
        coded_image[i] = transition
        
        start = transition

    #convert the coded image into a real image

    image = np.zeros((args.size**2, 3))

    for i, code in enumerate(coded_image):
        
        image[i] = colour_lookup(args.colors, code)

    image = image.reshape((args.size, args.size, 3))

    mpimg.imsave(f'art-{n}.png', image)

