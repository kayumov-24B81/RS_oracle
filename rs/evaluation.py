import os
import time
import torch
import numpy as np
from reedsolo import RSCodec, ReedSolomonError, rs_calc_syndromes
from rs.dataset_gen import bytes_to_bits, get_zero_mask

N, K, NSYM = 255, 223, 32

def predict_positions(model, noisy, threshold, device):
    syndrome = rs_calc_syndromes(noisy, NSYM)[1:]
    syndrome_bits = bytes_to_bits(bytes(syndrome))
    zero_mask = get_zero_mask(noisy)
    inp = np.concatenate([syndrome_bits, zero_mask])

    x = torch.tensor(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0].cpu().numpy()
    
    return [i for i in range(N) if probs[i] > threshold]

def evaluate_fsr(model, channel_fn, p_err, p_erase, threshold=0.3,
                 num_samples=1000, device='cpu'):
    model.eval()
    rsc = RSCodec(NSYM)
    success = 0

    for _ in range(num_samples):
        msg = os.urandom(K)
        codeword = rsc.encode(msg)
        noisy, _ = channel_fn(codeword, p_err, p_erase)

        erase_pos = predict_positions(model, noisy, threshold, device)

        try:
            decoded, _, _ = rsc.decode(noisy, erase_pos=erase_pos)
            if bytes(decoded) == msg:
                success += 1
        except ReedSolomonError:
            pass

    return success / num_samples

def compare_decoders(model, channel_fn, p_err, p_erase, threshold=0.3,
                     num_samples=1000, device='cpu'):
    model.eval()
    rsc = RSCodec(NSYM)

    classic_success = 0
    hybrid_success = 0
    hint_success = 0

    for _ in range(num_samples):
        msg = os.urandom(K)
        codeword = rsc.encode(msg)
        noisy, erase_pos = channel_fn(codeword, p_err, p_erase)

        try:
            decoded, _, _ = rsc.decode(noisy)
            if bytes(decoded) == msg:
                classic_success += 1
        except ReedSolomonError:
            pass

        try:
            decoded, _, _ = rsc.decode(noisy, erase_pos=erase_pos)
            if bytes(decoded) == msg:
                hint_success += 1
        except ReedSolomonError:
            pass

        pred_pos = predict_positions(model, noisy, threshold, device)
        try:
            decoded, _, _ = rsc.decode(noisy, erase_pos=pred_pos)
            if bytes(decoded) == msg:
                hybrid_success += 1
        except ReedSolomonError:
            pass
    
    return {
        'classic': classic_success / num_samples,
        'hybrid': hybrid_success / num_samples,
        'classic_hint': hint_success / num_samples,
    }

def benchmark_time(model, channel_fn, p_err, p_erase, threshold=0.3,
                   num_cycles=10, msgs_per_cycle=100, device='cpu'):
    model.eval()
    rsc = RSCodec(NSYM)

    classic_times = []
    hybrid_times = []

    for cycle in range(num_cycles):
        test_data = []
        for _ in range(msgs_per_cycle):
            msg = os.urandom(K)
            codeword = rsc.encode(msg)
            noisy, _ = channel_fn(codeword, p_err, p_erase)
            test_data.append(noisy)
        
        start = time.perf_counter()
        for noisy in test_data:
            try:
                rsc.decode(noisy)
            except ReedSolomonError:
                pass
        classic_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        for noisy in test_data:
            erase_pos = predict_positions(model, noisy, threshold, device)
            try:
                rsc.decode(noisy, erase_pos=erase_pos)
            except ReedSolomonError:
                pass
        hybrid_times.append(time.perf_counter() - start)

    return {
        'classic_mean': np.mean(classic_times),
        'classic_std': np.std(classic_times),
        'hybrid_mean': np.mean(hybrid_times),
        'hybrid_std': np.std(hybrid_times),
        'slowdown': np.mean(hybrid_times) / np.mean(classic_times),
        'msgs_per_cycle': msgs_per_cycle,
        'num_cycles': num_cycles,
    }

def full_evaluation(model, channel_fn, p_err, threshold=0.3, device='cpu'):
    p_erase_values = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
    results = []

    for p_erase in p_erase_values:
        r = compare_decoders(model, channel_fn, p_err, p_erase,
                             threshold=threshold, device=device)
        r['p_erase'] = p_erase
        results.append(r)

    return results