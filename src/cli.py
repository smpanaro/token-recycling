import argparse
from .token_recycling import TokenRecycling

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Token Recycling Text Generation')

    parser.add_argument(
        '--prompt',
        type=str,
        help='Input prompt for text generation',
        default="You are given two sorted lists of size m and n. Implement a function to find the kth smallest element in the union of the two lists with linear complexity."
        # default="This sequence seems to repeat forever: alpha beta gamma delta kappa.\n See: alpha beta gamma delta kappa alpha beta gamma delta kappa\n\nThe sequence of the first 1000 letters of the alphabet is called the alphabet of the alphabet."
    )

    parser.add_argument(
        '--model',
        type=str,
        default='HuggingFaceTB/SmolLM2-135M',
        help='Name or path of the model to use'
    )

    parser.add_argument(
        '--show-matrix',
        action='store_true',
        help='Print a sample of the adjacency matrix after generation'
    )

    args = parser.parse_args()

    recycler = TokenRecycling.from_pretrained(args.model)

    print(f"\nModel: {args.model}")
    print(args.prompt)
    recycler.generate(args.prompt)

    if args.show_matrix:
        recycler.print_matrix_sample()

if __name__ == '__main__':
    main()
