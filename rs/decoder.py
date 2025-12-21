from reedsolo import RSCodec, rs_calc_syndromes
from rs.dataset_gen import bytes_to_bits, get_zero_mask, RSPositionDataset
import torch
import numpy as np

class HybridDecoder:
    def __init__(self, model, threshold=0.3):
        self.model = model
        self.threshold = threshold
        self.rsc = RSCodec(32)
    
    def decode(self, noisy, device):
        syndrome = rs_calc_syndromes(noisy, 32)[1:]
        syndrome_bits = bytes_to_bits(bytes(syndrome))
        zero_mask = get_zero_mask(noisy)
        inp = np.concatenate([syndrome_bits, zero_mask]).astype(np.float32)
        
        x = torch.tensor(inp).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        
        positions = (probs[0] > self.threshold).cpu().numpy()
        erase_pos = [i for i, v in enumerate(positions) if v]
        
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except:
            return None