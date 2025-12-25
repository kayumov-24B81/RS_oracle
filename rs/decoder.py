from reedsolo import RSCodec, rs_calc_syndromes, ReedSolomonError
from rs.dataset_gen import bytes_to_bits, get_zero_mask, RSPositionDataset
import torch
import numpy as np

def compute_syndrome_bits(codeword, nsym=32):
    syndrome = rs_calc_syndromes(codeword, nsym)[1:]
    return bytes_to_bits(bytes(syndrome))

class HybridDecoder:
    def __init__(self, model, threshold=0.3, nsym=32, device='cpu'):
        self.model = model
        self.threshold = threshold
        self.nsym = nsym
        self.device = device
        self.rsc = RSCodec(nsym)
        
        self.model.eval()
        self.model.to(device)
    
    def predict_positions(self, noisy):
        syndrome_bits = compute_syndrome_bits(noisy, self.nsym)
        zero_mask = get_zero_mask(noisy)
        inp = np.concatenate([syndrome_bits, zero_mask])
        
        x = torch.tensor(inp).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.sigmoid(logits)
        
        positions = (probs[0] > self.threshold).cpu().numpy()
        return [i for i, v in enumerate(positions) if v]
    
    def decode(self, noisy):
        erase_pos = self.predict_positions(noisy)
        
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None

class ClassicDecoder:
    def __init__(self, nsym=32):
        self.nsym = nsym
        self.rsc = RSCodec(nsym)
    
    def decode(self, noisy, erase_pos=None):
        try:
            decoded, _, _ = self.rsc.decode(noisy, erase_pos=erase_pos)
            return bytes(decoded)
        except ReedSolomonError:
            return None
    
    def encode(self, message):
        return self.rsc.encode(message)

def evaluate_decoder(decoder, channel_fn, num_samples=1000, return_details=False):
    import os
    
    rsc = RSCodec(32)
    success = 0
    
    for _ in range(num_samples):
        msg = os.urandom(223)
        codeword = rsc.encode(msg)
        noisy, _ = channel_fn(codeword)
        
        decoded = decoder.decode(noisy)
        if decoded == msg:
            success += 1
    
    fsr = success / num_samples
    
    if return_details:
        return {
            'fsr': fsr,
            'success': success,
            'total': num_samples
        }
    return fsr

def compare_decoders(classic, hybrid, channel_fn, num_samples=1000):
    import os
    
    rsc = RSCodec(32)
    
    classic_success = 0
    hybrid_success = 0
    classic_hint_success = 0
    
    for _ in range(num_samples):
        msg = os.urandom(223)
        codeword = rsc.encode(msg)
        noisy, erase_pos = channel_fn(codeword)
        
        if classic.decode(noisy) == msg:
            classic_success += 1
        
        if hybrid.decode(noisy) == msg:
            hybrid_success += 1
        
        if classic.decode(noisy, erase_pos=erase_pos) == msg:
            classic_hint_success += 1
    
    return {
        'classic': classic_success / num_samples,
        'hybrid': hybrid_success / num_samples,
        'classic_hint': classic_hint_success / num_samples
    }
