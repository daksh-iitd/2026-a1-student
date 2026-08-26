"""
submission/stemmer.py — a from-scratch Porter stemmer + English stopword
list, used by submission/indexer.py's tokenize() so every scorer (BM25,
Boolean/VSM) sees the same normalized token stream.

Written from scratch rather than via NLTK: NLTK's Porter stemmer itself
needs no data download, but its `stopwords` corpus does (`nltk.download`),
and build_index()/load_index() must not require network access — the
grading container runs with `--network none` (docs/SUBMISSION_INTERFACE.md).
A self-contained implementation avoids that dependency entirely. The
assignment (Section 4.1) explicitly permits "standard libraries ... e.g.
NLTK's Porter stemmer", so either choice is allowed; this is the safer one
under the no-network constraint.

Algorithm: M. F. Porter, "An Algorithm for Suffix Stripping", Program,
14(3), 1980 — the same reference implicitly cited by the assignment spec.
Follows the paper's step 1a/1b/1c, 2, 3, 4, 5a/5b rule tables exactly,
including the m() "measure" (count of VC transitions), *v* (stem contains
a vowel), *d (ends in double consonant), and *o (ends cvc, not in wxy)
conditions.
"""
from functools import lru_cache
from typing import FrozenSet, List, Optional

# Standard SMART/NLTK English stopword list (179 words).
STOPWORDS: FrozenSet[str] = frozenset("""
i me my myself we our ours ourselves you youre youve youll youd your
yours yourself yourselves he him his himself she shes her hers herself
it its itself they them their theirs themselves what which who whom
this that thatll these those am is are was were be been being have has
had having do does did doing a an the and but if or because as until
while of at by for with about against between into through during before
after above below to from up down in out on off over under again further
then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very s t can will
just dont should shouldve now d ll m o re ve y ain aren arent
couldn couldnt didn didnt doesn doesnt hadn hadnt hasn hasnt haven
havent isn isnt ma mightn mightnt mustn mustnt needn neednt shan
shant shouldn shouldnt wasn wasnt weren werent won wont wouldn
wouldnt
""".split())


# Step 2 suffixes, ordered by descending length so a longer match (e.g.
# "ization") is always tried before a shorter one it also ends with
# (e.g. "ation") — see the module docstring in bm25.py's sibling files
# for why: the first matching suffix wins and stops the search.
_STEP2 = [
    ("ational", "ate"), ("ization", "ize"), ("iveness", "ive"),
    ("fulness", "ful"), ("ousness", "ous"), ("tional", "tion"),
    ("biliti", "ble"), ("entli", "ent"), ("ousli", "ous"),
    ("ation", "ate"), ("alism", "al"), ("aliti", "al"), ("iviti", "ive"),
    ("enci", "ence"), ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
    ("alli", "al"), ("ator", "ate"), ("eli", "e"),
]
_STEP3 = [
    ("icate", "ic"), ("ative", ""), ("alize", "al"), ("iciti", "ic"),
    ("ical", "ic"), ("ness", ""), ("ful", ""),
]
# (suffix, requires_s_or_t) — "ion" is special: it also requires the
# stem (after removing "ion") to end in 's' or 't'. All entries require
# m(stem) > 1; replacement is always "".
_STEP4 = [
    ("ement", False), ("ance", False), ("ence", False), ("able", False),
    ("ible", False), ("ment", False), ("ant", False), ("ion", True),
    ("ism", False), ("ate", False), ("iti", False), ("ous", False),
    ("ive", False), ("ize", False), ("ent", False), ("al", False),
    ("er", False), ("ic", False), ("ou", False),
]


class _PorterStemmer:
    def _is_consonant(self, word: str, i: int) -> bool:
        ch = word[i]
        if ch in "aeiou":
            return False
        if ch == "y":
            return i == 0 or not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem: str) -> int:
        """Porter's m(): the number of "vc" transitions in the
        consonant/vowel pattern of `stem`."""
        if not stem:
            return 0
        cv = "".join("c" if self._is_consonant(stem, i) else "v" for i in range(len(stem)))
        return cv.count("vc")

    def _contains_vowel(self, stem: str) -> bool:
        return any(not self._is_consonant(stem, i) for i in range(len(stem)))

    def _ends_double_consonant(self, word: str) -> bool:
        return (
            len(word) >= 2
            and word[-1] == word[-2]
            and self._is_consonant(word, len(word) - 1)
        )

    def _ends_cvc(self, word: str) -> bool:
        if len(word) < 3:
            return False
        i = len(word) - 1
        return (
            self._is_consonant(word, i - 2)
            and not self._is_consonant(word, i - 1)
            and self._is_consonant(word, i)
            and word[i] not in "wxy"
        )

    def _apply_first_match(self, word: str, rules, min_measure: int) -> str:
        for suffix, repl in rules:
            if word.endswith(suffix):
                stem = word[: len(word) - len(suffix)]
                if self._measure(stem) > min_measure:
                    return stem + repl
                return word
        return word

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2 or not word.isalpha():
            return word

        # Step 1a
        for suffix, repl in (("sses", "ss"), ("ies", "i"), ("ss", "ss"), ("s", "")):
            if word.endswith(suffix):
                word = word[: len(word) - len(suffix)] + repl
                break

        # Step 1b
        ed_or_ing_removed = False
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                word = stem + "ee"
        else:
            for suffix in ("ed", "ing"):
                if word.endswith(suffix):
                    stem = word[: -len(suffix)]
                    if self._contains_vowel(stem):
                        word = stem
                        ed_or_ing_removed = True
                    break

        if ed_or_ing_removed:
            if word.endswith(("at", "bl", "iz")):
                word += "e"
            elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif self._measure(word) == 1 and self._ends_cvc(word):
                word += "e"

        # Step 1c
        if word.endswith("y") and len(word) > 1 and self._contains_vowel(word[:-1]):
            word = word[:-1] + "i"

        # Step 2 & 3: longest-suffix-first, m(stem) > 0 required.
        word = self._apply_first_match(word, _STEP2, min_measure=0)
        word = self._apply_first_match(word, _STEP3, min_measure=0)

        # Step 4: m(stem) > 1 required; "ion" additionally needs the
        # stem to end in 's' or 't'.
        for suffix, requires_s_or_t in _STEP4:
            if word.endswith(suffix):
                stem = word[: len(word) - len(suffix)]
                if requires_s_or_t and not stem.endswith(("s", "t")):
                    continue
                if self._measure(stem) > 1:
                    word = stem
                break

        # Step 5a
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._ends_cvc(stem)):
                word = stem

        # Step 5b
        if (
            self._measure(word) > 1
            and self._ends_double_consonant(word)
            and word.endswith("l")
        ):
            word = word[:-1]

        return word


_stemmer = _PorterStemmer()


@lru_cache(maxsize=100_000)
def stem(word: str) -> str:
    """Cached: the same term recurs across many documents in a corpus, so
    memoizing avoids re-running the rule tables on every occurrence."""
    return _stemmer.stem(word)


def normalize_tokens(tokens: List[str], remove_stopwords: bool = True) -> List[str]:
    """Stopword-filter (optional) then stem a token list, in that order —
    stopwords are removed based on their surface form before stemming, so
    the stopword list above stays in plain-English form rather than
    needing to be pre-stemmed itself."""
    out: List[str] = []
    for tok in tokens:
        if remove_stopwords and tok in STOPWORDS:
            continue
        out.append(stem(tok))
    return out
