from typing import List, Dict

class Solution:
    def tokenize_numbers(self, numbers: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        out = []
        for num in numbers : 
            out.append(self.greedy_tokenize(str(num), vocab))
        return out
        pass

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        token_count = len(self.greedy_tokenize(text, vocab))
        return token_count 
        pass

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        token_count = self.count_tokens(text, vocab)
        word_count = len(text.split())
        print(token_count)
        print(word_count)
        return round(token_count / word_count, 4)
        pass

    def greedy_tokenize(self, num, vocab):
        i = 0
        out = []
        listnum = list(num)
        #if len(listnum) == 1
        while i < len(listnum):
            best = None
            for length in range(len(listnum), 0, -1):
                voc = "".join(listnum)[i:i+length]
                if voc in vocab:
                    best = voc
                    break
            if best is None:
                best = "".join(listnum)[0:1]
                    
            out.append(best)
            i+=length    
        return out 
