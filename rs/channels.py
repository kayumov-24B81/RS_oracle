import random
from typing import List, Tuple

def qsc_channel(codeword: bytes, p: float) -> bytes:
    """Simulates a q-ary symmetric channel (QSC).
    
    Each symbol of the input codeword is independently replaced
    with a random byte value with probability 'p'.

    Args:
        codeword: Input codeword as a byte sequence.
        p: Symbol error probability (0 <= p <= 1)

    Returns:
        A noisy version of the input codeword, wjere each symbol
        is independently corrupted with probability 'p'.
    """
    noisy = bytearray(codeword)
    for i in range(len(noisy)):
        if random.random() < p:
            noisy[i] = random.randint(0, 255)
    return bytes(noisy)

def qsc_erasure_channel(codeword, p_err, p_erase):
    """Simulates a QSC channel with erasures.
    
    Each symbol of the codeword undergoes one of the following events:
        - With probability 'p_erase', the symbol is erased (set to zero).
        - With probability 'p_err', the symbol is replaced by a random byte.
        - With probability '1 - p_erase - p_err', the symbol is unchanged.
    
    Erased symbols are explicitly marked by their indices.

    Args:
        codeword: Input codeword as a byte sequence.
        p_err: Probability of a symbol error.
        p_erase: Probability of a symbol erasure.
    
    Returns:
        A tuple (noisy, erasures):
            - noisy: Noisy codeword with errors and erasures applied.
            - erasures: List of indices corresponding to erased symbols.
    """
    noisy = bytearray(codeword)
    erasures = []

    for i in range(len(noisy)):
        r = random.random()
        if r < p_erase:
            noisy[i] = 0
            erasures.append(i)
        elif r < p_erase + p_err:
            noisy[i] = random.randint(0, 255)

    return bytes(noisy), erasures
