from collections import defaultdict
from itertools import chain, combinations


class FPNode:
    def __init__(self, item_name, frequency, parent_node):
        self.item_name = item_name
        self.count = frequency
        self.parent = parent_node
        self.children = {}
        self.next = None

    def increment(self, frequency):
        self.count += frequency


class FPGrowth:
    def __init__(self):
        pass

    @staticmethod
    def construct_tree(item_set_list, frequency, min_sup):
        header_table = defaultdict(int)
        # Counting frequency and create header table
        for idx, item_set in enumerate(item_set_list):
            for item in item_set:
                header_table[item] += frequency[idx]

        # Deleting items below minSup
        header_table = dict((item, sup) for item, sup in header_table.items() if sup >= min_sup)
        if len(header_table) == 0:
            return None, None

        # HeaderTable column [Item: [frequency, headNode]]
        for item in header_table:
            header_table[item] = [header_table[item], None]

        # Init Null head node
        fp_tree = FPNode('Null', 1, None)
        # Update FP tree for each cleaned and sorted item_set
        for idx, item_set in enumerate(item_set_list):
            item_set = [item for item in item_set if item in header_table]
            item_set.sort(key=lambda item: header_table[item][0], reverse=True)
            # Traverse from root to leaf, update tree with given item
            current_node = fp_tree
            for item in item_set:
                current_node = FPGrowth.update_tree(item, current_node, header_table, frequency[idx])

        return fp_tree, header_table

    @staticmethod
    def update_header_table(item, target_node, header_table):
        if header_table[item][1] is None:
            header_table[item][1] = target_node
        else:
            current_node = header_table[item][1]
            # Traverse to the last node then link it to the target
            while current_node.next is not None:
                current_node = current_node.next
            current_node.next = target_node

    @staticmethod
    def update_tree(item, tree_node, header_table, frequency):
        if item in tree_node.children:
            # If the item already exists, increment the count
            tree_node.children[item].increment(frequency)
        else:
            # Create a new branch
            new_item_node = FPNode(item, frequency, tree_node)
            tree_node.children[item] = new_item_node
            # Link the new branch to header table
            FPGrowth.update_header_table(item, new_item_node, header_table)

        return tree_node.children[item]

    @staticmethod
    def ascend_fptree(node, prefix_path):
        if node.parent is not None:
            prefix_path.append(node.item_name)
            FPGrowth.ascend_fptree(node.parent, prefix_path)

    @staticmethod
    def find_prefix_path(base_pat, header_table):
        # First node in linked list
        tree_node = header_table[base_pat][1]
        cond_pats = []
        frequency = []
        while tree_node is not None:
            prefix_path = []
            # From leaf node all the way to root
            FPGrowth.ascend_fptree(tree_node, prefix_path)
            if len(prefix_path) > 1:
                # Storing the prefix path and it's corresponding count
                cond_pats.append(prefix_path[1:])
                frequency.append(tree_node.count)

            # Go to next node
            tree_node = tree_node.next
        return cond_pats, frequency

    @staticmethod
    def mine_tree(header_table, min_sup, pre_fix, freq_item_list):
        # Sort the items with frequency and create a list
        sorted_item_list = [item[0] for item in sorted(list(header_table.items()), key=lambda p: p[1][0])]
        # Start with the lowest frequency
        for item in sorted_item_list:
            # Pattern growth is achieved by the concatenation of suffix pattern with frequent patterns generated from conditional FP-tree
            new_freq_set = pre_fix.copy()
            new_freq_set.add(item)
            freq_item_list.append(new_freq_set)
            # Find all prefix path, construct conditional pattern base
            conditional_patt_base, frequency = FPGrowth.find_prefix_path(item, header_table)
            # Construct conditional FP Tree with conditional pattern base
            conditional_tree, new_header_table = FPGrowth.construct_tree(conditional_patt_base, frequency, min_sup)
            if new_header_table is not None:
                # Mining recursively on the tree
                FPGrowth.mine_tree(new_header_table, min_sup,
                                   new_freq_set, freq_item_list)

    @staticmethod
    def powerset(s):
        return chain.from_iterable(combinations(s, r) for r in range(1, len(s)))

    @staticmethod
    def get_support(test_set, item_set_list):
        count = 0
        for item_set in item_set_list:
            if set(test_set).issubset(item_set):
                count += 1
        return count

    @staticmethod
    def association_rule(freq_item_set, item_set_list):
        rules = []
        for item_set in freq_item_set:
            subsets = FPGrowth.powerset(item_set)
            item_set_sup = FPGrowth.get_support(item_set, item_set_list)
            for s in subsets:
                rules.append([set(s), set(item_set.difference(s)), item_set_sup])
        return rules

    @staticmethod
    def get_frequency_from_list(item_set_list):
        frequency = [1 for _ in range(len(item_set_list))]
        return frequency

    @staticmethod
    def fpgrowth(item_set_list, min_sup_ratio):
        frequency = FPGrowth.get_frequency_from_list(item_set_list)
        # min_sup = len(item_set_list) * min_sup_ratio
        min_sup = min_sup_ratio
        fp_tree, header_table = FPGrowth.construct_tree(item_set_list, frequency, min_sup)
        if fp_tree is None:
            print('No frequent item set')
            return None, None
        else:
            freq_items = []
            FPGrowth.mine_tree(header_table, min_sup, set(), freq_items)
            rules = FPGrowth.association_rule(freq_items, item_set_list)

            freq_items_with_support = []
            for item_set in freq_items:
                item_set_sup = FPGrowth.get_support(item_set, item_set_list)
                freq_items_with_support.append([item_set_sup, item_set])

            return freq_items_with_support, rules
