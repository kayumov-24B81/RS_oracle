import os
from reedsolo import RSCodec
from channels import qsc_channel, qsc_erasure_channel

def count_symbol_errors(codeword, noisy, erasures):
    e = 0
    for i in range(len(codeword)):
        if i not in erasures and codeword[i] != noisy[i]:
            e += 1
    return e

def errors_only_baseline(p):
    rsc = RSCodec(32)
    msg = os.urandom(223)

    anomaly_count = 0

    e = count_symbol_errors(codeword, noisy, [])
    theory_check = e <= 32

    codeword = rsc.encode(223)
    noisy = qsc_channel(codeword, p)

    try:
        decoded, _, _ = rsc.decode(noisy)
        success = (decoded == msg)
    except:
        success = False

    if theory_check and not success:
        anomaly_count += 1
       
    frame_errors = int(not success)

    if success:
        bit_errors = sum(bin(a ^ b).count(1) for a, b in zip(decoded, msg))
    else:
        bit_errors = 8*223

    return frame_errors, bit_errors, anomaly_count

def errors_erasures_baseline(p_err, p_erase, DEBUG = False):
    rsc = RSCodec(32)
    msg = os.urandom(223)

    anomaly_count = 0

    codeword = rsc.encode(msg)
    noisy, erasures = qsc_erasure_channel(codeword, p_err, p_erase)

    s = len(erasures)
    e = count_symbol_errors(codeword, noisy, erasures)
    theory_check = (2*s + e <= 32)

    try:
        decoded, _, _ = rsc.decode(noisy, erase_pos=erasures)
        success = (decoded == msg)
    except:
        success = False
        pass

    if theory_check and not success:
        anomaly_count += 1
    
    frame_errors = int(not success)

    if success:
        bit_errors = sum(bin(a ^ b).count(1) for a, b in zip(decoded, msg))
    else:
        bit_errors = 8*223

    return frame_errors, bit_errors, anomaly_count


