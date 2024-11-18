import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.cache_utils import DynamicCache
import sys
from typing import Callable, TypeVar, List, Tuple, Optional, Union, cast
import time

from dataclasses import dataclass

try:
    from os_signpost import Signposter
    signposter = Signposter("com.stephenpanaro.tokenrecycling", Signposter.Category.DynamicTracing)
    # To record an Instruments trace:
    # xctrace record --template TraceTemplate.tracetemplate --launch -- env/bin/python -m src.cli
except ImportError:
    import contextlib
    class StubSignposter:
        def begin_interval(self, msg) -> Callable[[Optional[str]], None]: return lambda _: None
        @contextlib.contextmanager
        def use_interval(self, begin_msg: str, end_msg: Optional[str]=None): yield
        def emit_event(self, msg: str): pass
    signposter = StubSignposter()

@dataclass
class Config:
    # Hardcoded to match the static tree.
    matrix_top_k: int = 8  # Number of candidate tokens to store per token
    tree_depth: int = 6    # Maximum depth of the draft tree

@dataclass
class Outputs:
    output_ids: torch.Tensor
    accepted_sequences: List[List[int]]
    total_steps: int

class TokenRecycling:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        """
        Initialize TokenRecycling from a huggingface transformers model.
        """
        model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return cls(model, tokenizer)

    def __init__(self, model, tokenizer: PreTrainedTokenizer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = model.dtype
        self.model = model.to(self.device)
        self.tokenizer = tokenizer

        self.adjacency_matrix = torch.zeros(
            (self.tokenizer.vocab_size, Config.matrix_top_k),
            dtype=torch.long,
            device="cpu"
        )
        self.should_speculate = True
        self.use_cache = True # Use a KV cache.
        self.show_tokens = False # Print token IDs instead of decoding.

        # For this template, the root is the last verified token. We don't verify it but keep
        # it in the tree since it's easier to have a tree with a single root.
        # The next level is the predicted token -- we would get it even without speculating.
        # The 3rd level is where we start to get bonus speculated tokens.
        self.tree_template = self.static_tree()
        # Remove the root node from each.
        self.relative_position_ids = torch.tensor(self.get_relative_position_ids(self.tree_template), dtype=torch.long, device=self.device)[1:]
        self.tree_attention_mask = self.get_tree_attention_mask(self.tree_template, device=None).bool()[1:, 1:]

    def generate(self, prompt: Union[str, torch.Tensor], max_new_tokens=150, hot_start=False, silent=False):
        """
        Generate text using token recycling method.
        """
        input_ids = prompt.to(device=self.device) if isinstance(prompt, torch.Tensor) else \
             cast(torch.Tensor, self.tokenizer(prompt, return_tensors="pt").to(self.device).input_ids)
        prompt_length = input_ids.shape[-1]
        guess_length = 0 # Number of trailing tokens in input_ids that are guesses.
        total_accepted_tokens = 0
        total_guesses = 0
        past_key_values: Optional[DynamicCache] = DynamicCache() if self.use_cache else None
        prompt_start_time = time.time()
        generation_start_time = None
        accepted_seqs = []
        steps = 0

        if self.should_speculate:
            if not hot_start:
                self.adjacency_matrix.fill_(0)
            else:
                # TODO: Seed an initial guess.
                pass

        while input_ids.shape[-1] - prompt_length - guess_length < max_new_tokens:
            steps += 1
            with torch.no_grad():
                with signposter.use_interval("position_ids", "end"):
                    position_ids = self.get_position_ids(input_ids, guess_length)
                with signposter.use_interval("attention_mask", "end"):
                    attention_mask = self.get_attention_mask(input_ids, guess_length)
                use_full_input_ids = not self.use_cache or input_ids.shape[-1] == prompt_length
                with signposter.use_interval("forward", "end"):
                    logits = self.model(
                        input_ids if use_full_input_ids else input_ids[..., -(guess_length+1):],
                        position_ids=position_ids if use_full_input_ids or position_ids is None else position_ids[..., -(guess_length+1):],
                        attention_mask=attention_mask if use_full_input_ids or attention_mask is None else attention_mask[..., -(guess_length+1):, :],
                        past_key_values=past_key_values,
                        use_cache=self.use_cache,
                    ).logits

            if guess_length > 0:
                total_guesses += 1
            if generation_start_time is None:
                generation_start_time = time.time()

            sp = signposter.begin_interval("update matrix")
            next_token_index = -1 - guess_length
            if not hot_start:
                # Initialize from the entire prompt if cold-starting.
                self.adjacency_matrix[input_ids if use_full_input_ids else input_ids[..., -(guess_length+1):]] = logits.topk(Config.matrix_top_k).indices.to(self.adjacency_matrix.device)
                hot_start = True
            else:
                # Update the newest token and any guess tokens. Including guess tokens increases MAT.
                update_slice = next_token_index if guess_length == 0 else slice(next_token_index, None)
                self.adjacency_matrix[input_ids[:, update_slice]] = logits[:, update_slice, :].topk(Config.matrix_top_k).indices.to(self.adjacency_matrix.device)
            sp = sp("end")

            next_token = logits[:, next_token_index, :].argmax(dim=-1)

            if not silent:
                print(next_token.item() if self.show_tokens else self.tokenizer.decode(next_token), end=" " if self.show_tokens else "")
                sys.stdout.flush()

            if self.should_speculate:
                # Split inputs from guesses.
                input_length = input_ids.shape[-1] - guess_length # Avoid negative slicing when guess_length is 0.
                assert input_ids.shape[-1]  + next_token_index == input_length-1, f"Next token index {next_token_index} does not align with input length {input_length}"
                guesses = input_ids[..., input_length-1:] # NOTE: This includes a non-guess token, the root of the tree.
                input_ids = torch.cat([input_ids[... , :input_length], next_token.unsqueeze(0)], dim=-1)
                accepted_seqs.append([next_token.item()])

                # Get correct guesses, if any.
                if guess_length > 0:
                    # Look back 1 to include the newly-predicted token. All guesses are predicated on it.
                    guess_logits = logits[..., input_length-1:, :] if use_full_input_ids else logits
                    guess_max = guess_logits.argmax(dim=-1)
                    assert guess_max[0, 0] == next_token, f"Expected {next_token} as first part of guess but got {guess_max[0, 0]}"
                    with signposter.use_interval("longest sequence", "end"):
                        longest_sequence = self.get_longest_sequence(self.tree_template, guesses, guess_max)

                    if len(longest_sequence) > 0:
                        assert guesses[0, longest_sequence[1]] == next_token, f"Expected {next_token.item()} as first part of sequence but got {guesses[0, longest_sequence[0]]}"
                        verified_ids = guess_max[..., longest_sequence[..., 1:]]

                        if not silent:
                            colors = ["95", "94", "92", "93", "91"]
                            for idx, id in enumerate(verified_ids.squeeze(0)):
                                foreground = colors[idx % len(colors)]
                                ch = self.tokenizer.decode(id)
                                underline = ";4" if ch.isspace() else ""
                                val = id if self.show_tokens else ch
                                print(f"\033[{foreground}{underline}m{val}\033[0m", end=" " if self.show_tokens else "")
                                signposter.emit_event(f"accept {verified_ids.shape[-1]}")
                            sys.stdout.flush()

                        input_ids = torch.cat([input_ids, verified_ids], dim=-1)
                        with signposter.use_interval("cache update", "end"):
                            self.update_cache(past_key_values, guess_length, longest_sequence)
                        accepted_seqs[-1].extend(longest_sequence[1:].tolist())
                    elif past_key_values:
                        past_key_values.crop(input_length)

                # Generate new guesses.
                with signposter.use_interval("make guesses", "end"):
                    new_guesses = torch.tensor(self.merge_sequence(self.adjacency_matrix, self.tree_template, input_ids[..., -1]), device=self.device)[1:] # Exclude the latest known token.
                input_ids = torch.cat([input_ids, new_guesses.unsqueeze(0)], dim=-1)
                guess_length = new_guesses.shape[-1]
            else:
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=-1)

        if not silent:
            print("\n")
            if total_guesses > 0:
                print(f"Mean Accepted Tokens: {(torch.tensor([len(s) for s in accepted_seqs], dtype=torch.float).mean()):.2f}")
            if generation_start_time is not None:
                end_time = time.time()
                print(f"Prompt: {prompt_length / (generation_start_time-prompt_start_time):.2f} tokens/sec")
                print(f"Generation: {(input_ids.shape[-1] - prompt_length - guess_length) / (end_time-generation_start_time):.2f} tokens/sec")

        # Clean up outputs.
        input_ids = input_ids[..., :input_ids.shape[-1]-guess_length]

        # Strictly apply max token limit.
        if input_ids.shape[-1] > prompt_length + max_new_tokens:
            trim_length = input_ids.shape[-1] - prompt_length - max_new_tokens
            input_ids = input_ids[..., :-trim_length]
            while trim_length > 0:
                if len(accepted_seqs[-1]) <= trim_length:
                    trim_length -= len(accepted_seqs.pop())
                else:
                    accepted_seqs[-1] = accepted_seqs[-1][:-trim_length]
                    trim_length = 0
            assert sum([len(s) for s in accepted_seqs]) == input_ids.shape[-1] - prompt_length

        return Outputs(output_ids=input_ids, accepted_sequences=accepted_seqs, total_steps=steps)

    def get_position_ids(self, input_ids, guess_length: int) -> Optional[torch.Tensor]:
        if guess_length == 0:
            return None

        input_length = input_ids.shape[-1] - guess_length
        return torch.cat([
            torch.arange(input_length, dtype=self.relative_position_ids.dtype, device=self.device),
            self.relative_position_ids + input_length - 1
        ]).unsqueeze(0)

    def get_attention_mask(self, input_ids, guess_length: int) -> Optional[torch.Tensor]:
        if guess_length == 0:
            return None

        # Without KV Cache
        # ▉: Attend
        # ┌────────────────────────────┐
        # │▉                           │
        # │▉▉▉                         │
        # │▉▉▉▉▉                       │
        # │▉▉▉▉▉▉▉                     │
        # │▉▉▉▉▉▉▉▉▉                   │
        # │▉▉▉▉▉▉▉▉▉▉▉                 │
        # │▉▉▉▉▉▉▉▉Causal Mask         │
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉┌────────────┤
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉│▉           │
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉│▉▉▉         │
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉│▉Tree Mask  │
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉│▉ ▉   ▉     │
        # │▉▉▉▉▉▉▉▉▉▉▉▉▉▉▉│▉▉  ▉ ▉ ▉   │
        # └───────────────┴────────────┘

        # transformers expects 4D masks to be provided pre-inverted
        # for some attention implementations (eg SDPA supports bool tensors too)
        # 0: attend, large negative number: do not attend
        input_length = input_ids.shape[-1]
        min_dtype = torch.finfo(self.dtype).min
        mask = torch.full(
            (input_length, input_length), fill_value=min_dtype, dtype=self.dtype, device=self.device
        )
        mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

        if guess_length > 0:
            mask[:, :, -guess_length:, -guess_length:] = ((~self.tree_attention_mask.bool()) * min_dtype)
        return mask.to(dtype=self.dtype)

    def print_matrix_sample(self, num_samples=10):
        """
        Print a sample of the adjacency matrix for inspection.
        """
        print("\nAdjacency Matrix Sample:")
        print("Token ID | Token Text | Top-k Next Tokens")
        print("-" * 40)

        # Find non-zero rows
        non_zero_rows = (self.adjacency_matrix.sum(dim=1) != 0).nonzero().squeeze()

        # Print samples
        for token in non_zero_rows[:num_samples]:
            token_str = self.tokenizer.decode([token.item()])
            candidates = self.adjacency_matrix[token]
            candidate_strs = [self.tokenizer.decode([c.item()]) for c in candidates]
            print(f"{token:8d} | {token_str:10s} | {candidate_strs}")

    @classmethod
    def get_longest_sequence(cls, tree, guess_ids: torch.Tensor, actual_ids: torch.Tensor) -> torch.Tensor:
        """
        Compare the guessed speculated IDs to the actual IDs from the model output
        to see if any of the sequences were fully verified.

        tree: the top-k tree structure
        guess_ids: the input merged sequence [batch, guess_length+1], the tree root (not guessed) is in index 0
        actual_ids: the predicted merged sequence [batch, guess_length+1], actual[i] corresponds to the prediction to follow guess[i]
        """
        device = guess_ids.device
        guess_ids = guess_ids.cpu()
        actual_ids = actual_ids.cpu()

        def update(node, node_to_index, node_to_parent):
            node_to_index[node] = len(node_to_index)
            node_to_parent.update({child: node for child in node.children})

        node_to_index = {}
        node_to_parent = {}
        map_breadthfirst(tree, lambda node: update(node, node_to_index, node_to_parent))

        node_to_depth = {}
        stack = list(reversed(tree.children)) # Skip the root node.
        deepest = (None, -1)
        while stack:
            node = stack.pop()

            node_depth = node_to_depth.get(node, 0)
            node_to_depth.update({child: node_depth + 1 for child in node.children})

            guess_index = node_to_index.get(node, None)
            verify_index = node_to_index.get(node_to_parent.get(node), None)

            if not torch.equal(guess_ids[...,guess_index], actual_ids[...,verify_index]):
                # If current node is wrong, all children are too.
                continue

            if node_depth > deepest[1]:
                deepest = (node, node_depth)

            stack.extend(reversed(node.children))

        longest = []
        if deepest[0] is not None:
            curr = deepest[0]
            while curr is not None:
                idx = node_to_index.get(curr)
                curr = node_to_parent.get(curr, None)
                longest.append(idx)

        return torch.tensor(list(reversed(longest)), dtype=torch.long, device=device)

    @classmethod
    def update_cache(cls, cache: Optional[DynamicCache], guess_length: int, verified_indices: torch.Tensor):
        """
        Keep the entire non-guess part of the cache and add any verified parts.

        guess_length: 3
        verified_indices: [1]
        initial cache: [0 1 2 3 4 5 6 7 | 8 9 10]
        updated cache: [0 1 2 3 4 5 6 7 9]
        """
        if not cache:
            return

        length = cache.key_cache[0].shape[-2]
        input_length = cache.get_seq_length() - guess_length
        keep_indices = torch.cat([
            torch.arange(0, input_length, device=verified_indices.device),
            verified_indices[1:] - 1 + input_length
        ])
        for layer_idx, (k,v) in enumerate(zip(cache.key_cache, cache.value_cache)):
            cache.key_cache[layer_idx] = k.index_select(-2, keep_indices)
            cache.value_cache[layer_idx] = v.index_select(-2, keep_indices)

    @classmethod
    def get_relative_position_ids(cls, tree):
        """
        Generate the position ID offsets for the given tree. For instance:
             A
           ╱ │ ╲
          B  C  D
         ╱  ╱ ╲
        D  E   F
        │
        F
        The breadthwise merged sequence would be:
        [A, B, C, D, E, F, G]
        The relative position ids would be:
        [0, 1, 1, 1, 2, 2, 2, 3]
        """
        def get_depth(node: Tree, depths: dict[Tree, int]) -> int:
            depth = depths[node] + 1
            depths.update((child, depth) for child in node.children)
            return depths[node]

        depths = {tree: 0}
        return map_breadthfirst(tree, lambda node: get_depth(node, depths))

    @classmethod
    def get_tree_attention_mask(cls, tree, device=None) -> torch.Tensor:
        """
        Generate a 2D attention mask based on the given tree where
        each child only attends to itself and its ancestors.
           A
         ╱   ╲
        B     C
        │
        D
        The breadthwise merged sequence would be:
        [A, B, C, D]
        The attention mask would be [4,4]:
            A  B  C  D
        A [[1, 0, 0, 0],
        B  [1, 1, 0, 0],
        C  [1, 0, 1, 0],
        D  [0, 1, 0, 1]]
        """
        nodes = map_breadthfirst(tree, lambda x: x)
        n = len(nodes)

        node_to_index = {}
        node_to_parent = {}
        for i, node in enumerate(nodes):
            node_to_index[node] = i
            node_to_parent.update({child: node for child in node.children})

        mask = torch.zeros((n, n), dtype=torch.long, device=None)
        for i, node in enumerate(nodes):
            mask[i, i] = 1

            current = node
            while current in node_to_parent:
                parent = node_to_parent[current]
                parent_idx = node_to_index[parent]
                mask[i, parent_idx] = 1
                current = parent

        return mask

    @classmethod
    def merge_sequence(cls, M: torch.Tensor, tree, xt):
        """
        Make a merged sequence based on adjacency matrix M, static tree structure, and last prompt token xt.
        This is the flattened tree of guessed sequences that will be passed to the model.
        Algorithm 1: Static Tree Based BFS in the paper.
        """
        device = xt.device
        xt = xt.cpu()

        S = []  # 1: Initialize S ← ∅
        root = xt  # 2: Initialize root ← xt
        L = torch.tensor([root], dtype=torch.int, device=M.device) # 3: Initialize the current layer L ← (root)
        d = 0  # 4: Initialize the current depth d ← 0

        def get_all_tree_layers(tree):
            """
            Return a 3D list. [depth][node index][len(children)]
            """
            def update_layers(node, layers, node_to_depth):
                depth = node_to_depth[node] + 1
                node_to_depth.update((child, depth) for child in node.children)
                if len(layers) <= depth:
                    layers.append([])
                layers[depth].append([child.data for child in node.children])

            layers = [[tree.data]]
            node_to_depth = {tree: 0}
            map_breadthfirst(tree, lambda node: update_layers(node, layers, node_to_depth))
            return layers[:-1] # Drop the final empty layer.

        layer_indices = get_all_tree_layers(tree)[1:] # Skip first. Pre-compute this if it's slow.
        tree_depth = len(layer_indices)
        while d < tree_depth:  # 5: while d < Tree.depth do
            Lnext = []  # 6: Initialize next layer Lnext ← ∅

            # 7: Get all candidate tokens of L from M in parallel
            # 8: xs = M[L]
            xs = M[L] # tensor (# nodes in layer, K)

            # 9: Extract next layer tokens from xs with Tree
            # 10: Lnext = xs[Tree[d].index]
            Lnext = torch.cat([x[indices] for x, indices in zip(xs, layer_indices[d])])

            # 11: Concatenate S and L
            # 12: S ← (S;L)
            S.append(L)

            # 13: L ← Lnext
            L = Lnext
            d += 1

        S = torch.cat(S + [L])
        return S.tolist()  # 15: return S

    @staticmethod
    def static_tree():
        """ Static tree defined in Token Recycling Figure 5."""
        layers = [
            [[0, 1, 2, 3, 4, 5, 6, 7]],
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3], [0, 1, 2], [0, 1], [0], [0], [0], [0]],
            [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2], [0, 1], [0], [0], [0], [0], [0], [0, 1], [0], [], [], [0], [], [], [0], [], [0], [0], [], []],
            [[0, 1, 2, 3, 4], [0, 1], [0], [0], [0], [], [], [], [0], [], [], [0], [], [], [], [], [], [], [0], [], [], [0], [0], [], []],
            [[0, 1, 2], [0], [0], [], [], [0], [], [], [], [], [0], [], [0], [], []],
            [[0, 1], [], [], [0], [], [], [], []]
        ]

        root = Tree(data=0)
        curr = [root]
        for depth, layer in enumerate(layers):
            new_curr = []
            assert len(layer) == len(curr), f"Layer index {depth} does not match parent count, len(layer)={len(layer)}, len(curr)={len(curr)}"
            for children, parent in zip(layer, curr):
                parent.children = [Tree(data=c) for c in children]
                new_curr.extend(parent.children)
            curr = new_curr
        return root

class Tree:
    def __init__(self, data, children = []):
        self.children = children
        self.data = data

    def __repr__(self):
        return f"Tree({self.data}, {len(self.children)} children)"

T = TypeVar('T')
def map_breadthfirst(tree, fn: Callable[[Tree], T]) -> List[T]:
    """Map a function breadth-first over all nodes of a tree"""
    queue = [tree]
    res = []
    while queue:
        node = queue.pop(0)
        res.append(fn(node))
        queue.extend(node.children)
    return res

def map_depthfirst(tree, fn: Callable[[Tree], T]) -> List[T]:
    """Map a function depth-first over all nodes of a tree"""
    stack = [tree]
    res = []
    while stack:
        node = stack.pop()
        res.append(fn(node))
        stack.extend(reversed(node.children))
    return res
