import torch
import numpy as np
from tqdm import tqdm
from typing import Callable, Optional, Tuple, List, Dict, Tuple, Union
import string
import framework
import os
from .sequence import TextSequenceTestState, TextClassifierTestState


class CompositionalTableLookup:
    VERSION = 5
    cache = {}
    types = ["basic", "iid", "new_primitive_1", "new_primitive_2", "deeper", "deeper_val", "new_composition"]

    class CLTData:
        save = {"inputs", "outputs", "types", "indices", "version", "max_in_len", "max_out_len", "word_based"}
        version: int
        inputs: List[List[int]]
        outputs: List[List[int]]
        types: List[int]
        indices: Dict[str, List[int]]
        in_vocabulary: framework.data_structures.CharVocabulary
        out_vocabulary: framework.data_structures.CharVocabulary
        word_based: bool

        def construct(self, inputs: List[List[int]], outputs: List[List[int]], types: List[int],
                      indices: Dict[str, List[int]],
                      in_vocabulary: Union[framework.data_structures.CharVocabulary,
                                           framework.data_structures.WordVocabulary],
                      out_vocabulary: Union[framework.data_structures.CharVocabulary,
                                            framework.data_structures.WordVocabulary]):
            self.inputs = inputs
            self.outputs = outputs
            self.types = types
            self.indices = indices
            self.in_vocabulary = in_vocabulary
            self.out_vocabulary = out_vocabulary
            self.max_in_len = max(len(i) for i in inputs)
            self.max_out_len = max(len(i) for i in outputs)
            self.version = CompositionalTableLookup.VERSION
            self.word_based = isinstance(in_vocabulary, framework.data_structures.WordVocabulary)

            return self

        def state_dict(self):
            res = {k: self.__dict__[k] for k in self.save}
            res["in_vocabulary"] = self.in_vocabulary.state_dict()
            res["out_vocabulary"] = self.out_vocabulary.state_dict()
            return res

        def load_state_dict(self, state):
            self.__dict__.update({k: state[k] for k in self.save})
            if self.word_based:
                self.in_vocabulary = framework.data_structures.WordVocabulary(None)
                self.out_vocabulary = framework.data_structures.WordVocabulary(None)
            else:
                self.in_vocabulary = framework.data_structures.CharVocabulary(None)
                self.out_vocabulary = framework.data_structures.CharVocabulary(None)
            self.in_vocabulary.load_state_dict(state["in_vocabulary"])
            self.out_vocabulary.load_state_dict(state["out_vocabulary"])

    def get_spacer(self) -> str:
        return " " if self.atomic_input else ""

    def bit_format(self, arg: int) -> str:
        return f"{arg:08b}"[-self.n_bits:]

    def format_cmd_str(self, command, arg):
        spacer = self.get_spacer()
        if self.reversed:
            return spacer.join([self.bit_format(arg)] + [self.table_names[i] for i in reversed(command)])
        else:
            return spacer.join([self.table_names[i] for i in command] + [self.bit_format(arg)])

    def format_result(self, tables, command, arg):
        curr = arg
        if self.detailed_output:
            out = [self.bit_format(arg)] if self.copy_input else []
            for c in reversed(command):
                curr = tables[c][curr]
                out.append(self.bit_format(curr))
        else:
            for c in reversed(command):
                curr = tables[c][curr]
            out = [self.bit_format(curr)]

        return self.get_spacer().join(out)

    def sample_data(self, n_gen: int, tables: np.ndarray, sampler_fn: Callable[[], Tuple[int, List[int]]]) -> \
            Tuple[List[str], List[str], List[int]]:
        data = {}

        pbar = tqdm(total=n_gen)
      
        while len(data) < n_gen:
            arg, command = sampler_fn()
            cmd_str = self.format_cmd_str(command, arg)

            if cmd_str in data:
                continue

            pbar.update(1)
            data[cmd_str] = (self.format_result(tables, command, arg), len(command))

        inputs, outputs, depths = [], [], []
        for i, o in data.items():
            inputs.append(i)
            outputs.append(o[0])
            depths.append(o[1])

        return inputs, outputs, depths

    def generate_data(self) -> CLTData:
        s = np.random.RandomState(0x12345678)

        assert self.n_bits < 8

        tables = np.stack([s.permutation(2**self.n_bits) for p in range(self.n_tables)])
        tables_all_compositions = tables[:-2]

        n_possible_compositions = int(len(tables_all_compositions)**2 * \
                                  (len(tables_all_compositions)**(self.max_depth-1) - 1) / \
                                  (len(tables_all_compositions)-1))
        n_possible_sequences = int(2**self.n_bits * n_possible_compositions)

        print(f"Number of possible composite sequences: {n_possible_sequences}")
        print(f"Number of possible compositions: {n_possible_compositions}")

        # Reserve some compositions that will not be seen during the training
        test_compositions = set()
    
        for d in range(self.max_depth + 1):
            # Reserve max 25% of a given depth for testing novel compositions (max 200 per depth)
            n_this_depth = len(tables_all_compositions)**d
            n_reserved = min(200, n_this_depth * 0.25)
            n_before = len(test_compositions)
            while len(test_compositions) < n_before + n_reserved:
                test_compositions.add(tuple(s.randint(len(tables_all_compositions), size=[d]).tolist()))

        print(f"Number of reserved compositions: {len(test_compositions)}")

        n_iid_train = n_possible_sequences // 10
        n_iid_train = n_iid_train if self.max_n_samples is None else min(self.max_n_samples, n_iid_train)
        n_test = min(n_iid_train, 4000)

        inputs = []
        outputs = []
        types = []

        indices = {
            "train": [],
            "train_new_primitive": [],
            "valid": [],
            "test": [],
            "train_basic": [],
            "train_depth_2": [],
            "train_depth_3": [],
            "train_depth_4": []
        }

        t_index = {n: i for i, n in enumerate(self.types)}

        # Basic
        for t in range(self.n_tables):
            for a in range(2**self.n_bits):
                indices["train_basic"].append(len(inputs))
                indices["train"].append(len(inputs))
                indices["test"].append(len(inputs))
                inputs.append(self.format_cmd_str([t], a))
                outputs.append(self.format_result(tables, [t], a))
                types.append(t_index["basic"])

        # IID train, test, validation
        def sample_iid():
            while True:
                # Make sure it is not from the reserved list
                comp = tuple(s.randint(len(tables_all_compositions), size=[self.max_depth])
                             [:s.randint(2, self.max_depth + 1)].tolist())
                if comp not in test_compositions:
                    break

            return s.randint(2**self.n_bits), comp
                   
        i_iid, o_iid, depths = self.sample_data(n_iid_train + 2 * n_test, tables_all_compositions, sample_iid)
        offset = len(inputs)
        inputs += i_iid
        outputs += o_iid
        types += [t_index["iid"]] * len(i_iid)
        # Split randomly to train and test
        t_indices = set(np.random.choice(len(i_iid), 2*n_test, replace=False))
        for i in range(len(i_iid)):
            if i in t_indices:
                continue
            
            indices["train"].append(offset + i)
            if depths[i] in {2, 3, 4}:
                indices[f"train_depth_{depths[i]}"].append(offset + i)

        t_indices = list(t_indices)
        indices["test"] += [offset + i for i in t_indices[:n_test]]
        indices["valid"] += [offset + i for i in t_indices[n_test:]]

        # New primitive 1, test
        def sample_new_primitive_1():
            arg = s.randint(2**self.n_bits)
            command = s.randint(len(tables) - 1, size=[self.max_depth])[:s.randint(2, self.max_depth + 1)].tolist()
            command[s.randint(len(command))] = len(tables) - 2
            return arg, command

        i_new, o_new, _ = self.sample_data(1000, tables, sample_new_primitive_1)
        indices["test"] += [i + len(inputs) for i in range(len(i_new))]
        inputs += i_new
        outputs += o_new
        types += [t_index["new_primitive_1"]] * len(i_new)

        # New primitive 2, train, test
        def sample_new_primitive_2():
            arg = s.randint(2**self.n_bits)
            command = s.randint(len(tables) - 1, size=[self.max_depth])[:s.randint(2, self.max_depth + 1)]
            command[command == (len(tables) - 2)] = len(tables) - 1
            command[s.randint(len(command))] = len(tables) - 1
            return arg, command.tolist()

        i_new, o_new, _ = self.sample_data(3000, tables, sample_new_primitive_2)
        t_indices = set(np.random.choice(len(i_new), 1000, replace=False))
        offset = len(inputs)
        indices["train_new_primitive"] += [offset + i for i in range(len(i_new)) if i not in t_indices]
        indices["test"] += [offset + i for i in t_indices]
        inputs += i_new
        outputs += o_new
        types += [t_index["new_primitive_2"]] * len(i_new)

        # Deeper, valid
        def sample_deeper_valid():
            return s.randint(2**self.n_bits), \
                  s.randint(len(tables_all_compositions),
                             size=[self.max_depth + self.n_more_depth_valid])[:s.randint(self.max_depth + 1, \
                             self.max_depth + self.n_more_depth_valid + 1)].tolist()

        i_deeper, o_deeper, _ = self.sample_data(1000, tables_all_compositions, sample_deeper_valid)
        indices["valid"] += [i + len(inputs) for i in range(len(i_deeper))]
        inputs += i_deeper
        outputs += o_deeper
        types += [t_index["deeper_val"]] * len(i_deeper)
        
        # Deeper, test
        def sample_deeper():
            return s.randint(2**self.n_bits), \
                   s.randint(len(tables_all_compositions),
                             size=[self.max_depth + self.n_more_depth])[:s.randint(self.max_depth + 1 + self.n_more_depth_valid, \
                             self.max_depth + self.n_more_depth + 1)].tolist()

        i_deeper, o_deeper, _ = self.sample_data(1000, tables_all_compositions, sample_deeper)
        indices["test"] += [i + len(inputs) for i in range(len(i_deeper))]
        inputs += i_deeper
        outputs += o_deeper
        types += [t_index["deeper"]] * len(i_deeper)

        # New composition that is guarantueed to be not seen during training
        for com in tqdm(test_compositions):
            arg = s.randint(2**self.n_bits)
            indices["test"].append(len(inputs))
            inputs.append(self.format_cmd_str(com, arg))
            outputs.append(self.format_result(tables, com, arg))
            types.append(t_index["new_composition"])

        # Repeat the newly added primtive many times to have chance to actually remember it
        for i in range(self.n_tables - 2, self.n_tables):
            for _ in range(30):
                for a in range(2**self.n_bits):
                    indices["train"].append(len(inputs))
                    inputs.append(self.format_cmd_str([i], a))
                    outputs.append(self.format_result(tables, [i], a))
                    types.append(t_index["basic"])

        if self.atomic_input:
            bitsrs = [self.bit_format(b) for b in range(2**self.n_bits)]
            in_vocab = framework.data_structures.WordVocabulary(bitsrs + self.table_names)
            out_vocab = framework.data_structures.WordVocabulary(bitsrs)
        else:
            in_vocab = framework.data_structures.CharVocabulary(set(self.table_names) | {"0", "1"})
            out_vocab = framework.data_structures.CharVocabulary({"0", "1"})

        inputs = [in_vocab(i) for i in inputs]
        outputs = [out_vocab(o) for o in outputs]

        return self.CLTData().construct(inputs, outputs, types, indices, in_vocab, out_vocab)
     
    def __init__(self, split: Union[str, List[str]], max_depth: int, n_tables: int, n_bits: int = 3, n_more_depth: int = 3,
                 reversed: bool = False, cache: str = "./cache", max_n_sampes: Optional[int] = None,
                 atomic_input: bool = True, copy_input: bool = False, detailed_output: bool = True, n_more_depth_valid: int = 1) -> None:

        self.n_bits = n_bits
        self.max_depth = max_depth
        self.n_tables = n_tables
        self.n_bits = n_bits
        self.n_more_depth = n_more_depth
        self.n_more_depth_valid = n_more_depth_valid
        self.reversed = reversed
        self.max_n_samples = max_n_sampes
        self.atomic_input = atomic_input
        self.copy_input = copy_input
        self.detailed_output = detailed_output
        self.table_names = list(string.ascii_lowercase[:n_tables]) if len(string.ascii_lowercase) >= n_tables else \
                           [f"c{n+1}" for n in range(n_tables)]

        assert n_more_depth > n_more_depth_valid

        self.id = f"{max_depth}_{n_tables}_{n_bits}_{n_more_depth}_{int(reversed)}_{n_more_depth_valid}_"\
                  f"{self.max_n_samples}_{int(atomic_input)}_{int(copy_input)}_{detailed_output}"

        if self.id not in self.cache:
            cache_file = os.path.join(cache, self.__class__.__name__, self.id+".pth")
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)

            with framework.utils.LockFile(os.path.join(cache, self.__class__.__name__, "lock")):
                if os.path.isfile(cache_file):
                    self.cache[self.id] = self.CLTData()
                    self.cache[self.id].load_state_dict(torch.load(cache_file))

                if self.id not in self.cache or self.cache[self.id].version != self.VERSION:
                    self.cache[self.id] = self.generate_data()
                    torch.save(self.cache[self.id].state_dict(), cache_file)

            names = ", ".join(f"{len(v)} {k}" for k, v in self.cache[self.id].indices.items())
            print(f"{self.__class__.__name__}: Loaded {names}")

        self.data = self.cache[self.id]
        self.in_vocabulary = self.data.in_vocabulary
        self.out_vocabulary = self.data.out_vocabulary
        self.max_in_len = self.data.max_in_len
        self.max_out_len = self.data.max_out_len

        if isinstance(split, str):
            split = [split]

        self.indices = []
        for s in split:
            self.indices += self.data.indices[s]


    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item):
        i = self.indices[item]
        ins = self.data.inputs[i]
        outs = self.data.outputs[i]
        return {
            "in": np.asarray(ins, np.int8),
            "out": np.asarray(outs, np.int8),
            "in_len": len(ins),
            "out_len": len(outs),
            "type": self.data.types[i]
        }

    def start_test(self) -> TextSequenceTestState:
        if self.atomic_input:
            return TextSequenceTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                         lambda x: " ".join(self.out_vocabulary(x)),
                                         type_names = self.types)
        else:
            return TextSequenceTestState(self.in_vocabulary, self.out_vocabulary, type_names = self.types)


class CompositionalTableLookupClassification(CompositionalTableLookup): 
    def __init__(self, split: Union[str, List[str]], max_depth: int, n_tables: int, n_bits: int = 3,
                  n_more_depth: int = 3, reversed: bool = True, cache: str = "./cache",
                  max_n_sampes: Optional[int] = None, atomic_input: bool = True, n_more_depth_valid: int = 1) -> None:

        super().__init__(split, max_depth, n_tables, n_bits, n_more_depth, reversed=reversed, cache=cache,
                         max_n_sampes=max_n_sampes, atomic_input=atomic_input, copy_input=False, detailed_output=False,
                         n_more_depth_valid=n_more_depth_valid)

    def start_test(self) -> TextClassifierTestState:
        if self.atomic_input:
            return TextClassifierTestState(lambda x: " ".join(self.in_vocabulary(x)),
                                           lambda x: self.out_vocabulary([x])[0],
                                           type_names = self.types, max_good_samples=100, max_bad_samples=100)
        else:
            return TextClassifierTestState(self.in_vocabulary, lambda x: self.out_vocabulary([x])[0],
                                           type_names = self.types, max_good_samples=100, max_bad_samples=100)

    def __getitem__(self, item):
        i = self.indices[item]
        ins = self.data.inputs[i]
        outs = self.data.outputs[i]
        assert len(outs) == 1
        return {
            "in": np.asarray(ins, np.int8),
            "out": np.asarray(outs[0], np.int8),
            "in_len": len(ins),
            "out_len": len(outs),
            "type": self.data.types[i]
        }
