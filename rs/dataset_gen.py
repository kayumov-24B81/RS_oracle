from reedsolo import RSCodec, rs_calc_syndromes
from typing import Tuple
from rs.channels import qsc_erasure_channel
from torch.utils.data import Dataset
import numpy as np
import os


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert a byte sequence into a bit vector.
    
    Args:
        data: Input byte sequence
    
    Returns:
        A one-dimensional NumPy array of type uint8 containing
        bits (0 or 1). Length of the array is 8 * len(data).
    """
    arr = np.array(list(data), dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Converts a bit vector into a byte sequence.
    
    Args:
        bits: One-dimensional array of bits (0 or 1)
            The length must be a multuple of 8.
    Returns:
        bytes: Byte sequence obtained by packing the bits.
    """
    return bytes(np.packbits(bits))

def get_zero_mask(data: bytes) -> np.ndarray:
    """Creates a mask indicating zero-valued symbols.

    Positions with value zero are marked with 1.0, all others with 0.0

    Args:
        data: Input data sequence.
    
    Returns:
        A float32 NumPy array where:
            - 1.0 indicates a zero-valued symbol (erasure)
            - 0.0 indicates a non-zero symbol
    
    """
    return np.array([1.0 if b == 0 else 0.0 for b in data], dtype = np.float32)

class RSPositionDataset(Dataset):
    """Pytorch Dataset for learning symbol error positions in RS codes.

    Each dataset sample consists of:
        - Input features:
            * Bit-level representation of the RS syndrome
            * Zero-symbol (erasure) mask of the received codeword
        - Target labels:
            * Binary vector indicating symbol error locations
        
    The dataset simulates transmission throught a QSC erasure/error channel
    and is intented for supervised training of neural decoder.
    """

    def __init__(self, size: int, p_err: float, p_erase: float, nsym: int =32, msg_len: int = 223) -> None:
        """Initializes the dataset and generates samples.
        
        Args:
            size: Number of samples in the dataset.
            p_err: Probability of symbol error.
            p_erase : Probability of symbol erasure.
            nsym: Number of RS parity symbols.
            msg_len: Message length in bytes.
        """
        self.size: int = size
        self.p_err: float = p_err
        self.p_erase: float = p_erase
        self.nsym: int = nsym
        self.msg_len: int = msg_len
        self.n: int = msg_len + nsym

        self.rsc = RSCodec(nsym)

        self.inputs: np.ndarray
        self.positions: np.ndarray

        self._generate_data()

    def _generate_data(self) -> None:
        """Generates synthetic RS transmission data.
        
        For each sample:
            1. A random message is generated.
            2. The message is encoded using a RS code.
            3. The codeword is passed throught a QSC erasure/error channel.
            4. Syndromes are computed from the noisy codeword.
            5. Input features are constructed by concatenating:
                - Bit representation of the sydrome
                - Zero-symbol mask of the noisy codeword
            6. Target labels are created as a binary error-position vector.
        """
    
        inputs = []
        positions = []

        for _ in range(self.size):
            msg: bytes = os.urandom(self.msg_len)
            codeword: bytes = self.rsc.encode(msg)
            noisy, _ = qsc_erasure_channel(codeword, self.p_err, self.p_erase)

            syndrome = rs_calc_syndromes(noisy, self.nsym)[1:]
            syndrome_bits = bytes_to_bits(bytes(syndrome))
            zero_mask = get_zero_mask(noisy)
            input_vector = np.concatenate([syndrome_bits, zero_mask])

            error_pattern = bytes(a ^ b for a,b in zip(codeword, noisy))
            positions_vector = np.array([1.0 if e != 0 else 0.0 for e in error_pattern], dtype = np.float32)

            inputs.append(input_vector)
            positions.append(positions_vector)

        self.inputs = np.array(inputs, dtype=np.float32)
        self.positions = np.array(positions, dtype=np.float32)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.
        
        Returns:
            int: Dataset size.
        """
        return self.size
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Retrieves a single dataset sample.

        Args:
            idx (int): Sample index.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - input_vector: Float32 array containing syndrome bits
                  and the zero-symbol mask.
                - positions: Float32 binary array indicating error positions.
        """
        return self.inputs[idx], self.positions[idx]