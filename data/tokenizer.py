from typing import List
from collections import Counter

class Solution:
    def fusionner_couple(self, liste, couple):
        premier, second = couple
        resultat = []
        i = 0
        
        while i < len(liste):
            # Vérification dynamique du couple
            if i < len(liste) - 1 and liste[i] == premier and liste[i+1] == second:
                resultat.append(premier + second)
                i += 2  # Avance de 2 pour sauter les éléments fusionnés
            else:
                resultat.append(liste[i])
                i += 1  # Avance de 1 normalement
                
        return resultat

    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        char = list(corpus)
        merges = []
        for i in range(num_merges):
            compteur_paires = Counter(zip(char, char[1:]))
            dico = dict(compteur_paires)
            val_max = max(dico.values())
            cles = [cle for cle, val in dico.items() if val == val_max]
            to_merge = sorted(set(cles))[0]
            char = self.fusionner_couple(char, to_merge)
            merges.append(list(to_merge))
        return merges
        pass
