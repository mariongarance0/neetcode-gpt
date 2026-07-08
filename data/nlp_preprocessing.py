import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        if len(positive) != len(negative):
            raise ValueError("The two lists do not have the same length")
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
            
        positive_words = []
        negative_words = []
        all_words = []
        for i in range(len(positive)):
            positive_words = positive_words + positive[i].split()
            negative_words = negative_words + negative[i].split()
            #all_words = all_words + positive_words + negative_words

        all_words = positive_words + negative_words
        words_sorted = sorted(set(all_words))

        pos_nums = []
        neg_nums = []

        for i in range(len(positive)):
            positive_words = positive[i].split()
            negative_words = negative[i].split()
            pos_num = []
            for pos_word in positive_words:
                for order, word in enumerate(words_sorted):
                    if pos_word == word:
                        pos_num.append(order+1)
            pos_nums.append(pos_num)

            neg_num = []
            for neg_word in negative_words:
                for order, word in enumerate(words_sorted):
                    if neg_word == word:
                        neg_num.append(order+1)
                        break
            neg_nums.append(neg_num)

        all_nums = pos_nums + neg_nums
        tenseurs = [torch.tensor(sous_liste, dtype=torch.float32) for sous_liste in all_nums]

        tensor = nn.utils.rnn.pad_sequence(tenseurs, batch_first=True, padding_value=0.0)
        return tensor 
