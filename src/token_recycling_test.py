import torch
import unittest
from .token_recycling import TokenRecycling, Tree, map_depthfirst, map_breadthfirst

class TestTokenRecycling(unittest.TestCase):
    def test_merge_sequence(self):
        # Create a simple adjacency matrix: [token ID, top kth token ID]
        M = torch.tensor([
            [7, 2, 0, 5],
            [3, 6, 1, 4],
            [5, 2, 7, 0],
            [1, 6, 4, 3],
            [0, 7, 2, 4],
            [6, 3, 5, 1],
            [4, 1, 7, 2],
            [3, 0, 5, 6],
        ], dtype=torch.long)

        # Create a simple tree structure
        # Numbers are the Top K at that depth
        #       0
        #     / | \
        #    0  1  2
        #   /
        #  0
        root = Tree(0)
        root.children = [
            Tree(0, [Tree(0)]),
            Tree(1),
            Tree(2)
        ]

        # Set the last prompt token
        xt = 5

        #      5
        #    / | \
        #   6  3  5
        #  /
        # 4
        # Possible sequences: 5,6,4 and 5,3 and 5,5

        # Expected output sequence
        expected_sequence = [5, 6, 3, 5, 4]

        # Call the merge_sequence method
        result = TokenRecycling.merge_sequence(M, root, xt)

        # Assert that the result matches the expected sequence
        self.assertEqual(result, expected_sequence)

    def test_merge_sequence_empty_tree(self):
        M = torch.tensor([[1]], dtype=torch.long)
        root = Tree(0)
        xt = 0

        result = TokenRecycling.merge_sequence(M, root, xt)
        self.assertEqual(result, [0])

    def test_merge_sequence_large_matrix(self):
        # Create a larger adjacency matrix
        vocab_size = 128_256
        M = torch.randint(0, vocab_size, (vocab_size, 10), dtype=torch.int)

        # Create a more complex tree
        root = Tree(0)
        root.children = [
            Tree(1, [Tree(0), Tree(1)]),
            Tree(2, [Tree(0), Tree(1, [Tree(0)])]),
            Tree(3, [Tree(0, [Tree(1), Tree(2, [Tree(0)])])])
        ]

        xt = 0

        result = TokenRecycling.merge_sequence(M, root, xt)

        # one entry per tree node
        self.assertEqual(13, len(result))

        # first element is the root
        self.assertEqual(result[0], xt)

        # Check that all elements in the result are within the range of the matrix
        self.assertTrue(all(0 <= x < vocab_size for x in result))

    def test_map_breadth(self):
        root = Tree(0)
        root.children = [
            Tree(1, [Tree(3), Tree(4)]),
            Tree(2, [Tree(5)])
        ]

        visited = []
        map_breadthfirst(root, lambda node: visited.append(node.data))

        self.assertEqual(visited, [0, 1, 2, 3, 4, 5])

    def test_map_depth(self):
        root = Tree(0)
        root.children = [
            Tree(1, [Tree(3), Tree(4)]),
            Tree(2, [Tree(5)])
        ]

        visited = []
        map_depthfirst(root, lambda node: visited.append(node.data))

        self.assertEqual(visited, [0, 1, 3, 4, 2, 5])

    def test_get_relative_position_ids(self):
        # Test case 1: Simple tree
        #      0
        #    / | \
        #   1  2  3
        #  / \
        # 4   5
        root = Tree(0)
        root.children = [
            Tree(1, [Tree(4), Tree(5)]),
            Tree(2),
            Tree(3)
        ]

        relative_ids = TokenRecycling.get_relative_position_ids(root)
        expected_ids = [0, 1, 1, 1, 2, 2]
        self.assertEqual(relative_ids, expected_ids)

        # Test case 2: Single node tree
        single_node = Tree(0)
        relative_ids_single = TokenRecycling.get_relative_position_ids(single_node)
        expected_ids_single = [0]
        self.assertEqual(relative_ids_single, expected_ids_single)

        # Test case 3: Deep tree
        #    0
        #    |
        #    1
        #    |
        #    2
        #    |
        #    3
        deep_root = Tree(0)
        current = deep_root
        for i in range(3):
            new_node = Tree(i+1)
            current.children = [new_node]
            current = new_node

        relative_ids_deep = TokenRecycling.get_relative_position_ids(deep_root)
        expected_ids_deep = [0, 1, 2, 3]
        self.assertEqual(relative_ids_deep, expected_ids_deep)

        # Test case 4: Wide tree
        #       0
        #    / / \ \
        #   1 2  3  4
        wide_root = Tree(0)
        wide_root.children = [Tree(i) for i in range(1, 5)]

        relative_ids_wide = TokenRecycling.get_relative_position_ids(wide_root)
        expected_ids_wide = [0, 1, 1, 1, 1]
        self.assertEqual(relative_ids_wide, expected_ids_wide)

        # Test case 5: Complex tree
        #        0
        #      / | \
        #     1  2  3
        #    /  / \
        #   4  5   6
        #  /
        # 7
        complex_root = Tree(0)
        complex_root.children = [
            Tree(1, [Tree(4, [Tree(7)])]),
            Tree(2, [Tree(5), Tree(6)]),
            Tree(3)
        ]

        relative_ids_complex = TokenRecycling.get_relative_position_ids(complex_root)
        expected_ids_complex = [0, 1, 1, 1, 2, 2, 2, 3]
        self.assertEqual(relative_ids_complex, expected_ids_complex)

    def test_get_tree_attention_mask(self):
        # Test case 1: Simple tree
        #    A
        #  /   \
        # B     C
        # |
        # D
        root = Tree('A')
        root.children = [
            Tree('B', [Tree('D')]),
            Tree('C')
        ]

        mask = TokenRecycling.get_tree_attention_mask(root)
        expected_mask = torch.tensor([
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 0, 1]
        ], dtype=torch.long)

        self.assertTrue(torch.all(mask.eq(expected_mask)))

        # Test case 2: Single node tree
        single_node = Tree('A')
        single_mask = TokenRecycling.get_tree_attention_mask(single_node)
        expected_single_mask = torch.tensor([[1]], dtype=torch.long)

        self.assertTrue(torch.all(single_mask.eq(expected_single_mask)))

        # Test case 3: Deep tree
        #    A
        #    |
        #    B
        #    |
        #    C
        deep_root = Tree('A')
        deep_root.children = [Tree('B', [Tree('C')])]

        deep_mask = TokenRecycling.get_tree_attention_mask(deep_root)
        expected_deep_mask = torch.tensor([
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]
        ], dtype=torch.long)

        self.assertTrue(torch.all(deep_mask.eq(expected_deep_mask)))

        # Test case 4: Wide tree
        #       A
        #    / / \ \
        #   B C  D  E
        wide_root = Tree('A')
        wide_root.children = [Tree(c) for c in 'BCDE']

        wide_mask = TokenRecycling.get_tree_attention_mask(wide_root)
        expected_wide_mask = torch.tensor([
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 0, 1]
        ], dtype=torch.long)

        self.assertTrue(torch.all(wide_mask.eq(expected_wide_mask)))

        # Test case 5: Complex tree
        #        A
        #      / | \
        #     B  C  D
        #    /  / \
        #   E  F   G
        #  /
        # H
        complex_root = Tree('A')
        complex_root.children = [
            Tree('B', [Tree('E', [Tree('H')])]),
            Tree('C', [Tree('F'), Tree('G')]),
            Tree('D')
        ]

        complex_mask = TokenRecycling.get_tree_attention_mask(complex_root)
        expected_complex_mask = torch.tensor([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 0, 0, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [1, 0, 1, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1]
        ], dtype=torch.long)

        self.assertTrue(torch.all(complex_mask.eq(expected_complex_mask)))

    def test_get_longest_sequence(self):
        # Figure 2 in the paper.
        #           [guest]
        #    [speaker]    [speak]
        #   [at]  [for]      [ings]
        tree = Tree("guest")
        tree.children = [
            Tree("speaker", [Tree("at"), Tree("for")]),
            Tree("speak", [Tree("ings")])
        ]

        #            |- last known token + tree root
        #            v
        # inputs: [guest   speaker speak at for ings]
        # preds : [speaker at      ers   a  a   :]
        #
        # tree index   0        1        2      3    4    5
        # guess_ids:  [guest,   speaker, speak, at,  for, ings]
        #             11        12       13     14   15   16
        # actual_ids: [speaker, at,      ers,   a,   a,   :   ]
        #              12       14       17     18   18   19
        # expected: [1, 3]

        guess_ids  = torch.tensor([[11, 12, 13, 14, 15, 16]])
        actual_ids = torch.tensor([[12, 14, 17, 18, 18, 19]])

        longest = TokenRecycling.get_longest_sequence(tree, guess_ids, actual_ids)
        expected = torch.tensor([0, 1, 3], dtype=torch.long)
        self.assertTrue(torch.equal(longest, expected))


if __name__ == '__main__':
    unittest.main()
