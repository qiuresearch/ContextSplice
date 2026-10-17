# ---- Helper Functions ----
import numpy as np

# ---- One-hot encoding dictionary ----
BASE_DICT = {'A': [1, 0, 0, 0],
             'C': [0, 1, 0, 0],
             'G': [0, 0, 1, 0],
             'T': [0, 0, 0, 1],
             'N': [0, 0, 0, 0]}

def onehot_encode_CL(seq): # CL --> return in shape [C, L]
    return np.array([BASE_DICT.get(base.upper(), [0, 0, 0, 0]) for base in seq]).T


def decode_one_hot_old(array):
    index_to_base = {tuple([1, 0, 0, 0]): 'A',
                     tuple([0, 1, 0, 0]): 'C',
                     tuple([0, 0, 1, 0]): 'G',
                     tuple([0, 0, 0, 1]): 'T',
                     tuple([0, 0, 0, 0]): 'N'}  # unknown or padding

    return ''.join(index_to_base.get(tuple(vec), 'N') for vec in array.T)


def decode_onehot(onehot_vec, dim=0, idx2base=np.array(['A', 'C', 'G', 'T', 'N'])):
    """ Decode one-hot encoded vector to string sequence.
    Args:
        onehot_vec: np.ndarray, one-hot encoded array
        dim: int, dimension representing bases
        idx2base: np.ndarray, mapping from index to base, with the last representing all zeros
    """
    onehot_idx = np.argmax(onehot_vec, axis=dim)
    onehot_idx[np.sum(onehot_vec, axis=dim) == 0] = len(idx2base) - 1  # set 'N' for all-zero columns
    return ''.join(idx2base[onehot_idx])


def reverse_complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq))


def sprinkle_sites_on_sequence(gene_sites, n_classes=4, icol=None, debug=False):
    """ Convert splice site positions and labels into vectors aligned with the RNA sequence """

    seq = gene_sites['seq']
    # classification labels: 0: no site, 1: donor, 2: acceptor, 3: hybrid
    seq_cls = np.zeros(len(seq), dtype=int)
    seq_cls_odds = np.zeros(len(seq), dtype=float)
    seq_cls_mask = np.zeros(len(seq), dtype=int)
    # pos is 0-based, pretty slow way to do this
    for pos, site_type, site_odds in zip(gene_sites['cls_pos'], gene_sites['cls_type'], gene_sites['cls_odds']):
        if icol is not None:
            site_type = site_type[icol]
            site_odds = site_odds[icol]
        
        seq_cls_odds[pos] = site_odds
        seq_cls_mask[pos] = 1
        if site_type == 'donor':
            seq_cls[pos] = 1 # donor
        elif site_type == 'acceptor':
            seq_cls[pos] = 2 # acceptor
        elif site_type == 'hybrid':
            seq_cls[pos] = 3 # hybrid
        elif site_type.upper() in ['NONE', 'NULL', 'UNK', 'UNKNOWN']:
            seq_cls[pos] = 0 # no site
        else:
            raise ValueError(f"Invalid site type: {site_type}")

    # psi values (initialize with zeros!)
    seq_psi = np.zeros(len(seq), dtype=float)
    seq_psi_std = np.zeros(len(seq), dtype=float)
    seq_psi_mask = np.zeros(len(seq), dtype=int)
    # pos is 0-based, pretty slow way to do this
    for pos, psi, psi_std in zip(gene_sites['psi_pos'], gene_sites['psi'], gene_sites['psi_std']):
        if icol is not None:
            psi = psi[icol]
            psi_std = psi_std[icol]
        seq_psi[pos] = psi if psi is not None and not np.isnan(psi) else 0.0
        seq_psi_std[pos] = psi_std
        seq_psi_mask[pos] = 1

    return {'seq': seq,
            'cls': seq_cls if n_classes > 3 else np.clip(seq_cls, 0, n_classes - 1),  # binary classification if n_classes <= 3
            'cls_odds': seq_cls_odds,
            'cls_mask': seq_cls_mask,
            'psi': seq_psi,
            'psi_std': seq_psi_std,
            'psi_mask': seq_psi_mask,
            }


def get_crop_feats_from_single_seq(seq_feat, num_crops=None, 
                                   crop_size=5000, flank_size=5000,
                                   min_sites=0, min_usage=0,
                                   debug=False):
    """ Generate crop features for a single RNA transcript by croping the sequence
    Args:
        rna_feats: dict, output of sprinkle_sites_onto_vectors
        num_crops: int, number of crops to generate, if None, generate all possible crops
        crop_size: int, size of the crop (excluding flanking regions)
        flank_size: int, size of the flanking regions on each side
        min_sites: int, minimum number of splice sites in the crop to be included
        min_usage: float, minimum total usage (psi) in the crop to be included
        debug: bool, whether to print debug information
    """
    seq = seq_feat['seq']
    # dtypes set to be correct for model input
    # caution: lower precisions used for saving .h5 files whenever possible
    seq_cls = seq_feat['cls'].astype(np.int64)
    seq_cls_odds = seq_feat['cls_odds'].astype(np.float32)
    seq_cls_mask = seq_feat['cls_mask'].astype(np.int32)
    seq_psi = seq_feat['psi'].astype(np.float32)
    seq_psi_std = seq_feat['psi_std'].astype(np.float32)
    seq_psi_mask = seq_feat['psi_mask'].astype(np.int32)

    total_size = crop_size + 2 * flank_size

    # divide gene_list into crops of crop_size
    seq_len = len(seq)
    crop_starts = np.arange(0, seq_len, crop_size, dtype=np.int32)
    crop_starts[-1] = max([0, seq_len - crop_size])  # ensure last crop reaches the end

    crop_ends = crop_starts + crop_size
    crop_ends[-1] = seq_len  # ensure last crop reaches the end

    if debug:
        print(f"Seq Length: {seq_len}\ncrop starts: {crop_starts}\ncrop ends: {crop_ends}")

    # randomly pick num_crops pairs of crop_starts and crop_ends
    if num_crops is not None and num_crops > 0 and num_crops < len(crop_starts):
        selected_indices = np.random.choice(len(crop_starts), size=num_crops, replace=False)
        crop_starts = crop_starts[selected_indices]
        crop_ends = crop_ends[selected_indices]
        
    crop_feats = []
    for crop_start, crop_end in zip(crop_starts, crop_ends):
        # crop_end - crop_start <= crop_size always (< if gene is shorter than crop_size)
        crop_feat = {'crop_start': crop_start, 'crop_end': crop_end}

        # seq_start and seq_end include flanking regions
        seq_start = max(0, crop_start - flank_size)
        seq_end = min(seq_len, crop_end + flank_size)

        pad_size = total_size - (seq_end - seq_start)
        left_pad = flank_size - (crop_start - seq_start)
        right_pad = pad_size - left_pad
        
        # pad with 'N's if needed for input X
        crop_feat['seq'] = 'N' * left_pad + seq[seq_start:seq_end] + 'N' * right_pad
        crop_feat['seq_onehot'] = onehot_encode_CL(crop_feat['seq']).astype(np.float32)  # shape [4, total_size]

        crop_feat['cls'] = np.pad(seq_cls[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=0)
        crop_feat['cls_odds'] = np.pad(seq_cls_odds[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=0.)
        crop_feat['cls_mask'] = np.pad(seq_cls_mask[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=-100)

        crop_feat['psi'] = np.pad(seq_psi[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=0.)
        crop_feat['psi_std'] = np.pad(seq_psi_std[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=0.)
        crop_feat['psi_mask'] = np.pad(seq_psi_mask[seq_start:seq_end], (left_pad, right_pad), mode='constant', constant_values=-100)

        # target Y only has the middle crop_size region excluding flanks
        crop_feat['cls'] = crop_feat['cls'][flank_size:flank_size + crop_size]
        crop_feat['cls_odds'] = crop_feat['cls_odds'][flank_size:flank_size + crop_size]
        crop_feat['cls_mask'] = crop_feat['cls_mask'][flank_size:flank_size + crop_size] # not used currently
        crop_feat['psi'] = crop_feat['psi'][flank_size:flank_size + crop_size]
        crop_feat['psi_std'] = crop_feat['psi_std'][flank_size:flank_size + crop_size]
        crop_feat['psi_mask'] = crop_feat['psi_mask'][flank_size:flank_size + crop_size] # not used currently

        # parts of Y labels may be padding if the entire sequence is shorter than crop_size
        # train_feat['crop_mask'] = np.zeros(crop_size, dtype=np.int32)
        # the actual start of the seq[start:end] within the crop_size region of seq
        # y_start = left_pad + (crop_start - seq_start) - flank_size
        # y_end = y_start + (crop_end - crop_start)
        # train_feat['crop_mask'][y_start:y_end] = 1

        if min_sites and np.sum(crop_feat['cls'][:crop_end-crop_start]) <= min_sites:
            # print(f"Skipping gene_id: {train_feat['gene_id']}; start: {train_feat['start']}; end: {train_feat['end']} with <= {min_sites} splice sites!")
            continue

        if min_usage and np.sum(crop_feat['psi'][:crop_end-crop_start]) < min_usage:
            # print(f"Skipping gene_id: {train_feat['gene_id']}; start: {train_feat['start']}; end: {train_feat['end']} with <= {min_usage} usage!")
            continue

        crop_feats.append(crop_feat)

    return crop_feats


def make_blocks_no_Pad(start, end, chrom, genome, PADDING=5000, BLOCK_SIZE = 15000):
    region_start = max(0, start - PADDING)
    region_end = end + PADDING
    sequence = genome[chrom][region_start:region_end].seq.upper()
    
    blocks = []
    for i in range(0, len(sequence) - BLOCK_SIZE + 1, PADDING):
        block_seq = sequence[i:i + BLOCK_SIZE]
        blocks.append((region_start + i, region_start + i + BLOCK_SIZE, block_seq))
    return blocks


def make_blocks_Pad(start, end, chrom, genome, PADDING=5000, BLOCK_SIZE=15000):
    ### pad make blocks, very important for 
    ### if region < block_size
    
    region_start = max(0, start - PADDING)
    region_end = end + PADDING
    sequence = genome[chrom][region_start:region_end].seq.upper()

    blocks = []
    if len(sequence) < BLOCK_SIZE:
        # Pad with Ns to reach BLOCK_SIZE
        pad_len = BLOCK_SIZE - len(sequence)
        padded_seq = sequence + 'N' * pad_len
        blocks.append((region_start, region_start + BLOCK_SIZE, padded_seq))
    else:
        # Standard sliding window
        for i in range(0, len(sequence) - BLOCK_SIZE + 1, PADDING):
            block_seq = sequence[i:i + BLOCK_SIZE]
            blocks.append((region_start + i, region_start + i + BLOCK_SIZE, block_seq))
    return blocks


def make_blocks(start, end, chrom, genome, FIXED_PADDING=5000, BLOCK_SIZE=15000):
    """
    Create blocks of sequence for a region. Uses dynamic padding if region is short;
    otherwise uses fixed padding and sliding window.

    Args:
        start (int): Start coordinate of region.
        end (int): End coordinate of region.
        chrom (str): Chromosome name.
        genome (pyfaidx.Fasta): Loaded genome object.
        BLOCK_SIZE (int): Desired block size.
        FIXED_PADDING (int): Padding to apply for long regions.

    Returns:
        List of (block_start, block_end, sequence) tuples.
    """
    
    region_len = abs(end - start)

    if region_len <= BLOCK_SIZE:
        # Case 1: Short region → apply dynamic padding
        pad_total = max(0, BLOCK_SIZE - region_len)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left

        region_start = max(0, start - pad_left)
        region_end = end + pad_right

        sequence = genome[chrom][region_start:region_end].seq.upper()
        if len(sequence) < BLOCK_SIZE:
            sequence += 'N' * (BLOCK_SIZE - len(sequence))

        return [(region_start, region_end, sequence)]

    else:
        # Case 2: Long region → fixed padding and sliding window
        region_start = max(0, start - FIXED_PADDING)
        region_end = end + FIXED_PADDING

        full_seq = genome[chrom][region_start:region_end].seq.upper()
        blocks = []

        for i in range(0, len(full_seq) - BLOCK_SIZE + 1, FIXED_PADDING):
            block_seq = full_seq[i:i + BLOCK_SIZE]
            block_start = region_start + i
            block_end = block_start + BLOCK_SIZE
            blocks.append((block_start, block_end, block_seq))

        return blocks


def assign_labels(block_start, block_end, psi_dict, PADDING=5000, BLOCK_SIZE = 15000):
    labels = np.zeros((BLOCK_SIZE, 3))  # 12 for 1 tissue (spliced/unspliced/usage) x 4 tissues
    labels[:, 0] = 1 ## default unspliced site for all
    mid_start = PADDING
    mid_end = BLOCK_SIZE - PADDING

    for pos, psi in psi_dict.items():
        if block_start <= pos < block_end:
            rel_pos = pos - block_start
            if 0 <= rel_pos < BLOCK_SIZE:
                # For this example: tissue is heart → positions 0–2
                labels[rel_pos, 0] = 1 if psi < 0.1 else 0
                labels[rel_pos, 1] = 0 if psi < 0.1 else 1
                labels[rel_pos, 2] = 0 if psi < 0.1 else psi
    #return labels[mid_start:mid_end]  # Return only the middle 5000
    return labels.T  # Return Full


