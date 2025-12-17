from reedsolo import RSCodec, rs_calc_syndromes
from rs.channels import qsc_channel
import numpy as np
import os
import random

def bytes_to_bits(data):
    arr = np.array(list(data), dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bites(bits):
    return bytes(np.packbits(bits))

class RSDataset:
    def __init__(self, nsym=32, msg_len=223):
        self.nsym = nsym
        self.msg_len = msg_len
        self.n = msg_len + nsym # 255
        self.rsc = RSCodec(nsym)
    
    def generate_sample(self, p):
        msg = os.urandom(self.msg_len)
        codeword = self.rsc.encode(msg)
        noisy = qsc_channel(codeword, p)
        syndrome = rs_calc_syndromes(noisy, self.nsym)[1:]

        error_pattern = bytes(a ^ b for a, b in zip(codeword, noisy))

        syndrome_bits = bytes_to_bits(bytes(syndrome))
        error_bits = bytes_to_bits(error_pattern)

        return syndrome_bits, error_bits
    
    def generate_batch(self, batch_size, p):
        syndromes = []
        errors = []

        for _ in range(batch_size):
            s, e = self.generate_sample(p)
            syndromes.append(s)
            errors.append(e)
        
        return np.array(syndromes, dtype=np.float32), np.array(errors, dtype=np.float32)
    
dataset = RSDataset()
X, y = dataset.generate_batch(4, p=0.05)

print(f'Форма входа (синдромы): {X.shape}')
print(f'Форма выхода (error patterns): {y.shape}')
print(f'\nПример синдрома (первые 16 бит): {X[0, :16].astype(int)}')
print(f'Пример error pattern (первые 16 бит): {y[0, :16].astype(int)}')

errors_per_sample = [np.sum(y[i].reshape(-1, 8).any(axis=1)) for i in range(4)]
print(f'\nКоличество ошибочных символов в каждом примере: {errors_per_sample}')