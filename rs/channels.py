import random

def qsc_channel(codeword, p):
    noisy = bytearray(codeword)
    for i in range(len(noisy)):
        if random.random() < p:
            noisy[i] = random.randint(0, 255)
    return bytes(noisy)

def qsc_erasure_channel(codeword, p_err, p_erase):
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
