import argparse
import numpy as np
import matplotlib.colors
import matplotlib.image as mpimg

parser = argparse.ArgumentParser(description='Generate simple "art" from a Markov chain')
parser.add_argument('-s', type=int, default=256,
                    help='size of the image')
parser.add_argument('-c', nargs='+', default=['b', 'k', 'w'],
                    help='which colours to use, available: [\'b\', \'g\', \'r\', \'c\', \'m\', \'y\', \'k\', \'w\']')
parser.add_argument('-i', type=int, default=300,
                    help='how much extra inertia to have on a colour when generating that colour')
parser.add_argument('-n', type=int, default=1,
                    help='how many colours get the extra intertia specified by -i')
parser.add_argument('-g', type=int, default=10,
                    help='how many images to generate')
args = parser.parse_args()

#making sure colours are correct

available_colours = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

colours = args.c

for c in colours:
    if c not in available_colours:
        raise ValueError(f'Colours must be one of {available_colours}, you gave: {c}')

#generating transition probabilities

for n in range(args.g):

    transition_probs = np.ones((len(colours),len(colours)))

    for i in transition_probs:
        for _ in range(args.n):
            idx = np.random.randint(len(i))
            i[idx] = args.i

    #normalize transition probabilities

    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    normalised_transition_probs = softmax(transition_probs)

    #generate the "coded" image, each pixel is an integer representing a colour

    def colour_lookup(colours, code):
        colour = colours[int(code)]
        return np.array(matplotlib.colors.to_rgb(colour))

    coded_image = np.zeros(args.s**2)

    start = np.random.randint(len(colours))

    for i, _ in enumerate(coded_image):
        
        transition = np.random.choice(len(colours), p=normalised_transition_probs[:,start])
        
        coded_image[i] = transition
        
        start = transition

    #convert the coded image into a real image

    image = np.zeros((args.s**2, 3))

    for i, code in enumerate(coded_image):
        
        image[i] = colour_lookup(colours, code)

    image = image.reshape((args.s, args.s, 3))

    mpimg.imsave(f'art-{n}.png', image)

