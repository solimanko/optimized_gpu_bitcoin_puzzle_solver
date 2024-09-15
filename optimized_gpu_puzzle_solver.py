import numpy as np
import base58
import pyopencl as cl
import bitcoin
from bitcoin import *
import time
import json
from datetime import datetime
from tqdm import tqdm
import math
import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

# Base58 alphabet
BASE58_ALPHABET = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'

def base58_encode(b):
    n = int.from_bytes(b, 'big')
    chars = []
    while n:
        chars.append(BASE58_ALPHABET[n % 58])
        n //= 58
    return ''.join(reversed(chars))

def get_opencl_context():
    platforms = cl.get_platforms()
    if not platforms:
        raise RuntimeError("No OpenCL platforms found")
    
    print("Available OpenCL platforms:")
    for i, platform in enumerate(platforms):
        print(f"  [{i}] {platform.name}")
        devices = platform.get_devices()
        for j, device in enumerate(devices):
            print(f"    [{j}] {device.name} (Type: {cl.device_type.to_string(device.type)})")
    
    # Try to get a GPU device
    for platform in platforms:
        devices = platform.get_devices(device_type=cl.device_type.GPU)
        if devices:
            print(f"Using GPU device: {devices[0].name}")
            return cl.Context(devices)
    
    # If no GPU is available, fall back to CPU
    print("No GPU devices found, falling back to CPU")
    return cl.Context(cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU))

def create_opencl_program(ctx, opencl_code):
    try:
        program = cl.Program(ctx, opencl_code)
        program.build(options=['-cl-std=CL1.2'])
        return program
    except cl.RuntimeError as e:
        print(f"OpenCL build error: {e}")
        print(program.get_build_info(ctx.devices[0], cl.program_build_info.LOG))
        raise

def save_progress(start_range, current_range, end_range, total_checked, elapsed_time):
    progress = {
        "timestamp": datetime.now().isoformat(),
        "start_range": hex(start_range),
        "current_range": hex(current_range),
        "end_range": hex(end_range),
        "total_checked": total_checked,
        "elapsed_time": elapsed_time,
        "keys_per_second": total_checked / elapsed_time if elapsed_time > 0 else 0
    }
    with open("search_progress.json", "w") as f:
        json.dump(progress, f, indent=2)
        
def bsgs_search(target_address, start_range, end_range, batch_size=1000, max_baby_steps=100000):
    try:
        ctx = get_opencl_context()
        queue = cl.CommandQueue(ctx)
        program = create_opencl_program(ctx, opencl_code)
        start_time = time.time()
        total_checked = 0
        last_save_time = start_time

        # Convert target address to binary format
        target_address_bin = base58.b58decode_check(target_address)

        # Calculate BSGS parameters
        range_size = end_range - start_range
        m = min(int(math.sqrt(range_size)) + 1, max_baby_steps)  # Limit baby steps size

        # Generate baby steps in smaller chunks
        baby_step_chunk_size = min(m, batch_size)  # Process batch_size baby steps at a time

        goals = [0.04, 0.2125, 0.46]
        goal_keys = [int(goal * range_size) for goal in goals]
        goal_reached = [False, False, False]

        with tqdm(total=range_size, unit='keys', unit_scale=True, dynamic_ncols=True) as pbar:
            for giant_step in range(start_range, end_range, m):
                for baby_step_start in range(0, m, baby_step_chunk_size):
                    baby_step_end = min(baby_step_start + baby_step_chunk_size, m)
                    current_baby_steps = np.arange(baby_step_start, baby_step_end, dtype=np.uint64)
                    d_baby_steps = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=current_baby_steps)

                    current_start = giant_step + baby_step_start
                    current_end = min(current_start + len(current_baby_steps), end_range)
                    current_batch_size = current_end - current_start

                    d_found = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=np.int32().nbytes)
                    d_result = cl.Buffer(ctx, cl.mem_flags.READ_WRITE, size=np.uint64().nbytes * 2)

                    cl.enqueue_fill_buffer(queue, d_found, np.int32(0), 0, np.int32().nbytes)

                    program.bsgs_search(queue, (len(current_baby_steps),), None,
                                        np.uint64(giant_step & 0xFFFFFFFFFFFFFFFF),
                                        np.uint64(giant_step >> 64),
                                        d_baby_steps,
                                        np.uint64(len(current_baby_steps)),
                                        np.uint64(current_batch_size),
                                        cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=target_address_bin),
                                        d_found,
                                        d_result)

                    h_found = np.empty(1, dtype=np.int32)
                    cl.enqueue_copy(queue, h_found, d_found)

                    if h_found[0]:
                        h_result = np.empty(2, dtype=np.uint64)
                        cl.enqueue_copy(queue, h_result, d_result)
                        result = (h_result[1] << 64) | h_result[0]
                        return hex(result)[2:].zfill(64)

                    total_checked += current_batch_size
                    pbar.update(current_batch_size)
                    elapsed_time = time.time() - start_time
                    keys_per_second = total_checked / elapsed_time
                    bsgs_speed = keys_per_second * m

                    # Check if we've reached any of our goals
                    for i, goal in enumerate(goal_keys):
                        if not goal_reached[i] and total_checked >= goal:
                            goal_reached[i] = True
                            print(f"\nReached {goals[i]*100}% of the search in {elapsed_time/3600:.2f} hours")

                    # Inside the main loop of bsgs_search function
                    elapsed_time = time.time() - start_time
                    keys_per_second = total_checked / elapsed_time if elapsed_time > 0 else 0
                    bsgs_speed = keys_per_second * m

                    # Calculate remaining keys and time estimates
                    remaining_keys = range_size - total_checked
                    time_to_completion = remaining_keys / bsgs_speed if bsgs_speed > 0 else float('inf')

                    # Calculate time to reach specific percentages
                    time_to_4_percent = max(0, (0.04 * range_size - total_checked) / bsgs_speed) if bsgs_speed > 0 else float('inf')
                    time_to_21_25_percent = max(0, (0.2125 * range_size - total_checked) / bsgs_speed) if bsgs_speed > 0 else float('inf')
                    time_to_46_percent = max(0, (0.46 * range_size - total_checked) / bsgs_speed) if bsgs_speed > 0 else float('inf')

                    pbar.set_postfix({
                     'Current': f'0x{giant_step+baby_step_start:x}',
                     'Speed': f'{keys_per_second:.2f} keys/s',
                     'BSGS': f'{bsgs_speed:.2f} keys/s',
                     'To 4%': f'{time_to_4_percent/3600:.2f}h',
                     'To 21.25%': f'{time_to_21_25_percent/3600:.2f}h',
                     'To 46%': f'{time_to_46_percent/3600:.2f}h',
                      'ETA': f'{time_to_completion/3600:.2f}h'  
                    })


                    # Save progress every 5 minutes
                    if time.time() - last_save_time > 300:
                        save_progress(start_range, current_start, end_range, total_checked, elapsed_time)
                        last_save_time = time.time()

        return None, total_checked  # Return None for no result, and total_checked
    except cl.RuntimeError as e:
        print(f"OpenCL error: {e}")
        return None, total_checked
    except KeyboardInterrupt:
        print("\nSearch interrupted. Saving progress...")
        save_progress(start_range, giant_step+baby_step_start, end_range, total_checked, time.time() - start_time)
        return None, total_checked
    
opencl_code = """
#define ROTRIGHT(a,b) (((a) >> (b)) | ((a) << (32-(b))))
#define CH(x,y,z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x,y,z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTRIGHT(x,2) ^ ROTRIGHT(x,13) ^ ROTRIGHT(x,22))
#define EP1(x) (ROTRIGHT(x,6) ^ ROTRIGHT(x,11) ^ ROTRIGHT(x,25))
#define SIG0(x) (ROTRIGHT(x,7) ^ ROTRIGHT(x,18) ^ ((x) >> 3))
#define SIG1(x) (ROTRIGHT(x,17) ^ ROTRIGHT(x,19) ^ ((x) >> 10))

__constant uint k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};

__constant char base58_alphabet[] = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

typedef struct {
    uint x[8];
    uint y[8];
} Point;

__constant uint FIELD_PRIME[8] = {0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};
__constant uint CURVE_ORDER[8] = {0xD0364141, 0xBFD25E8C, 0xAF48A03B, 0xBAAEDCE6, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF};

void sha256_transform(uint* state, const uchar* block) {
    uint a, b, c, d, e, f, g, h, i, j, t1, t2, m[64];
    for (i = 0, j = 0; i < 16; ++i, j += 4)
        m[i] = (block[j] << 24) | (block[j + 1] << 16) | (block[j + 2] << 8) | (block[j + 3]);
    for ( ; i < 64; ++i)
        m[i] = SIG1(m[i - 2]) + m[i - 7] + SIG0(m[i - 15]) + m[i - 16];
    a = state[0];
    b = state[1];
    c = state[2];
    d = state[3];
    e = state[4];
    f = state[5];
    g = state[6];
    h = state[7];
    for (i = 0; i < 64; ++i) {
        t1 = h + EP1(e) + CH(e,f,g) + k[i] + m[i];
        t2 = EP0(a) + MAJ(a,b,c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

void sha256(const uchar* data, uint len, uchar* hash) {
    uint state[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };
    uint i, bitlen[2];
    uchar block[64];
    for (i = 0; i < len; ++i) {
        block[i % 64] = data[i];
        if ((i + 1) % 64 == 0) {
            sha256_transform(state, block);
        }
    }
    i = len % 64;
    block[i++] = 0x80;
    if (i > 56) {
        while (i < 64)
            block[i++] = 0x00;
        sha256_transform(state, block);
        i = 0;
    }
    while (i < 56)
        block[i++] = 0x00;
    bitlen[0] = len * 8;
    bitlen[1] = 0;
    for (i = 0; i < 8; ++i)
        block[56 + i] = (bitlen[i / 4] >> (24 - 8 * (i % 4))) & 0xFF;
    sha256_transform(state, block);
    for (i = 0; i < 8; ++i) {
        hash[i * 4] = (state[i] >> 24) & 0xFF;
        hash[i * 4 + 1] = (state[i] >> 16) & 0xFF;
        hash[i * 4 + 2] = (state[i] >> 8) & 0xFF;
        hash[i * 4 + 3] = state[i] & 0xFF;
    }
}

void ripemd160(const uchar* data, uint len, uchar* hash) {
    // Implement RIPEMD160 here
    // This is a placeholder implementation
    for (int i = 0; i < 20; i++) {
        hash[i] = data[i % len];
    }
}

void mod_add(uint* result, const uint* a, const uint* b) {
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        uint sum = a[i] + b[i] + carry;
        result[i] = sum;
        carry = sum < a[i] || (sum == a[i] && carry);
    }
    if (carry || (result[7] > FIELD_PRIME[7] || (result[7] == FIELD_PRIME[7] && result[6] > FIELD_PRIME[6]))) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = result[i] - FIELD_PRIME[i] - carry;
            result[i] = diff;
            carry = result[i] > diff;
        }
    }
}

void mod_sub(uint* result, const uint* a, const uint* b) {
    uint borrow = 0;
    for (int i = 0; i < 8; i++) {
        uint diff = a[i] - b[i] - borrow;
        result[i] = diff;
        borrow = (a[i] < b[i]) || (a[i] == b[i] && borrow);
    }
    if (borrow) {
        uint carry = 0;
        for (int i = 0; i < 8; i++) {
            uint sum = result[i] + FIELD_PRIME[i] + carry;
            result[i] = sum;
            carry = sum < result[i];
        }
    }
}

void mod_mul(uint* result, const uint* a, const uint* b) {
    ulong t[16] = {0};
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            t[i+j] += (ulong)a[i] * b[j];
        }
    }
    for (int i = 15; i > 7; i--) {
        t[i-8] += t[i] * 977;
        t[i-7] += t[i] >> 32;
        t[i-6] += t[i] & 0xFFFFFFFF;
    }
    uint carry = 0;
    for (int i = 0; i < 8; i++) {
        result[i] = t[i] + carry;
        carry = result[i] < t[i];
    }
    if (carry || (result[7] > FIELD_PRIME[7] || (result[7] == FIELD_PRIME[7] && result[6] > FIELD_PRIME[6]))) {
        carry = 0;
        for (int i = 0; i < 8; i++) {
            uint diff = result[i] - FIELD_PRIME[i] - carry;
            result[i] = diff;
            carry = result[i] > diff;
        }
    }
}

void point_add(Point* result, const Point* p, const Point* q) {
    uint u[8], v[8], z[8], t[8];
    mod_sub(u, q->y, p->y);
    mod_sub(v, q->x, p->x);
    mod_mul(z, u, u);
    mod_mul(t, v, v);
    mod_mul(u, t, v);
    mod_sub(z, z, t);
    mod_sub(t, p->x, q->x);
    mod_mul(v, t, z);
    mod_sub(t, u, v);
    mod_sub(u, t, p->x);
    mod_sub(v, p->y, u);
    mod_mul(z, v, t);
    mod_sub(result->x, z, p->y);
    mod_sub(result->y, u, q->x);
}

void scalar_mul(Point* result, const uint* k, const Point* p) {
    Point r = *p;
    for (int i = 255; i >= 0; i--) {
        point_add(&r, &r, &r);
        if ((k[i/32] >> (i%32)) & 1) {
            point_add(&r, &r, p);
        }
    }
    *result = r;
}

void base58_encode(const uchar* input, int input_len, __global char* output) {
    // This is a simplified placeholder implementation
    // In a real scenario, you'd need a proper base58 encoding algorithm
    for (int i = 0; i < input_len && i < 34; i++) {
        output[i] = base58_alphabet[input[i] % 58];
    }
    output[input_len < 34 ? input_len : 34] = 0;  // Null terminator
}

__kernel void bsgs_search(const ulong range_start_low,
                          const ulong range_start_high,
                          __global const ulong* baby_steps,
                          const ulong baby_steps_size,
                          const ulong batch_size,
                          __global const uchar* target_address,
                          __global int* found,
                          __global ulong* result) {
    size_t idx = get_global_id(0);
    if (idx >= baby_steps_size) return;
    
    ulong private_key_low = range_start_low + baby_steps[idx];
    ulong private_key_high = range_start_high + (private_key_low < range_start_low);
    
    Point G = {{0x16F81798, 0x59F2815B, 0x2DCE28D9, 0x029BFCDB, 0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E},
               {0xFB10D4B8, 0x9C47D08F, 0x68554199, 0xFD17B448, 0x0E1108A8, 0x5DA4FBFC, 0x26A3C465, 0x483ADA77}};
    
    for (ulong i = 0; i < baby_steps_size; i++) {
        ulong current_private_key_low = private_key_low + baby_steps[i];
        ulong current_private_key_high = private_key_high + (current_private_key_low < private_key_low);
        
        uint private_key_array[8];
        private_key_array[0] = current_private_key_low & 0xFFFFFFFF;
        private_key_array[1] = (current_private_key_low >> 32) & 0xFFFFFFFF;
        private_key_array[2] = current_private_key_high & 0xFFFFFFFF;
        private_key_array[3] = (current_private_key_high >> 32) & 0xFFFFFFFF;
        private_key_array[4] = 0;
        private_key_array[5] = 0;
        private_key_array[6] = 0;
        private_key_array[7] = 0;
        
        Point public_key;
        scalar_mul(&public_key, private_key_array, &G);
        
        uchar serialized_pubkey[65];
        serialized_pubkey[0] = 0x04;  // Uncompressed public key
        for (int j = 0; j < 8; j++) {
            serialized_pubkey[1 + j*4]     = (public_key.x[7-j] >> 24) & 0xFF;
            serialized_pubkey[1 + j*4 + 1] = (public_key.x[7-j] >> 16) & 0xFF;
            serialized_pubkey[1 + j*4 + 2] = (public_key.x[7-j] >> 8) & 0xFF;
            serialized_pubkey[1 + j*4 + 3] = public_key.x[7-j] & 0xFF;
        }
        for (int j = 0; j < 8; j++) {
            serialized_pubkey[33 + j*4]     = (public_key.y[7-j] >> 24) & 0xFF;
            serialized_pubkey[33 + j*4 + 1] = (public_key.y[7-j] >> 16) & 0xFF;
            serialized_pubkey[33 + j*4 + 2] = (public_key.y[7-j] >> 8) & 0xFF;
            serialized_pubkey[33 + j*4 + 3] = public_key.y[7-j] & 0xFF;
        }
        
        uchar sha256_result[32];
        sha256(serialized_pubkey, 65, sha256_result);
        
        uchar ripemd160_result[20];
        ripemd160(sha256_result, 32, ripemd160_result);
        
        uchar versioned_hash[21];
        versioned_hash[0] = 0x00;  // Mainnet version
        for (int j = 0; j < 20; j++) {
            versioned_hash[j + 1] = ripemd160_result[j];
        }
        
        uchar checksum[32];
        sha256(versioned_hash, 21, checksum);
        sha256(checksum, 32, checksum);
        
        uchar binary_address[25];
        for (int j = 0; j < 21; j++) {
            binary_address[j] = versioned_hash[j];
        }
        for (int j = 0; j < 4; j++) {
            binary_address[21 + j] = checksum[j];
        }
        
        bool match = true;
        for (int j = 0; j < 25; j++) {
            if (binary_address[j] != target_address[j]) {
                match = false;
                break;
            }
        }
        
        if (match) {
            atomic_cmpxchg(found, 0, 1);
            result[0] = current_private_key_low;
            result[1] = current_private_key_high;
            return;
        }
    }
}
"""

def main():
    # Define the full range
    full_start_range = int('20000000000000000', 16)
    full_end_range = int('3ffffffffffffffff', 16)
    
    # Calculate the size of each part
    full_range_size = full_end_range - full_start_range
    
    # Calculate the 20% that has already been searched
    already_searched = int(full_range_size * 0.2)
    
    # Define our actual search range
    start_range = full_start_range + already_searched
    end_range = full_end_range
    
    # Target Bitcoin address
    target_address = '13zb1hQbWVsc2S7ZTZnP2G4undNNpdh5so'
    
    print(f"Searching for private key for address: {target_address}")
    print(f"Search range: {hex(start_range)} to {hex(end_range)}")
    print(f"Already searched: {already_searched} keys (20% of full range)")
    
    # Search for the key
    start_time = time.time()
    result, keys_checked = bsgs_search(target_address, start_range, end_range, batch_size=1000, max_baby_steps=100000)
    end_time = time.time()

    # Calculate and print progress
    elapsed_time = end_time - start_time
    total_checked = already_searched + keys_checked
    keys_per_second = keys_checked / elapsed_time if elapsed_time > 0 else 0
    effective_keys_per_second = total_checked / elapsed_time if elapsed_time > 0 else 0
    progress_percentage = (total_checked / full_range_size) * 100

    if result:
        print(f"Found matching private key: {result}")
        
        # Verify the result
        pubkey = bitcoin.privkey_to_pubkey(result)
        address = bitcoin.pubkey_to_address(pubkey)
        if address == target_address:
            print("Verification successful! The private key generates the correct address.")
        else:
            print("Verification failed. The generated address does not match the target.")
    else:
        print("No matching private key found in the given range.")
    
    print(f"\nSearch Statistics:")
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"Keys checked in this run: {keys_checked:,}")
    print(f"Total keys checked (including already searched): {total_checked:,}")
    print(f"Speed of this run: {keys_per_second:.2f} keys/s")
    print(f"Effective speed (including already searched): {effective_keys_per_second:.2f} keys/s")
    print(f"Progress: {progress_percentage:.2f}%")

    # Calculate time estimates
    remaining_keys = full_range_size - total_checked
    time_to_completion = remaining_keys / keys_per_second if keys_per_second > 0 else float('inf')
    time_to_4_percent = max(0, (0.04 * full_range_size - total_checked) / keys_per_second) if keys_per_second > 0 else float('inf')
    time_to_21_25_percent = max(0, (0.2125 * full_range_size - total_checked) / keys_per_second) if keys_per_second > 0 else float('inf')
    time_to_46_percent = max(0, (0.46 * full_range_size - total_checked) / keys_per_second) if keys_per_second > 0 else float('inf')

    print(f"\nTime Estimates (based on current run speed):")
    print(f"Estimated time to completion: {time_to_completion/3600:.2f} hours")
    print(f"Estimated time to 4%: {time_to_4_percent/3600:.2f} hours")
    print(f"Estimated time to 21.25%: {time_to_21_25_percent/3600:.2f} hours")
    print(f"Estimated time to 46%: {time_to_46_percent/3600:.2f} hours")

    # Save final progress
    save_progress(start_range, start_range + keys_checked, end_range, total_checked, elapsed_time)

if __name__ == "__main__":
    main()
