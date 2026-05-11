import os

datasets = {
    'QuickDraw': {
        'photo_train': 'datasets/QuickDraw/zeroshot/all_photo_train.txt',
        'photo_test': 'datasets/QuickDraw/zeroshot/all_photo_zero.txt',
        'sketch_train': 'datasets/QuickDraw/zeroshot/sketch_train.txt',
        'sketch_test': 'datasets/QuickDraw/zeroshot/sketch_zero.txt',
    },
    'Sketchy_zeroshot0': {
        'photo_train': 'datasets/Sketchy/zeroshot0/all_photo_filelist_train.txt',
        'photo_test': 'datasets/Sketchy/zeroshot0/all_photo_filelist_zero.txt',
        'sketch_train': 'datasets/Sketchy/zeroshot0/sketch_tx_000000000000_ready_filelist_train.txt',
        'sketch_test': 'datasets/Sketchy/zeroshot0/sketch_tx_000000000000_ready_filelist_zero.txt',
    },
    'Sketchy_zeroshot1': {
        'photo_train': 'datasets/Sketchy/zeroshot1/all_photo_filelist_train.txt',
        'photo_test': 'datasets/Sketchy/zeroshot1/all_photo_filelist_zero.txt',
        'sketch_train': 'datasets/Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_train.txt',
        'sketch_test': 'datasets/Sketchy/zeroshot1/sketch_tx_000000000000_ready_filelist_zero.txt',
    },
    'TUBerlin': {
        'photo_train': 'datasets/TUBerlin/zeroshot/ImageResized_ready_filelist_train.txt',
        'photo_test': 'datasets/TUBerlin/zeroshot/ImageResized_ready_filelist_zero.txt',
        'sketch_train': 'datasets/TUBerlin/zeroshot/png_ready_filelist_train.txt',
        'sketch_test': 'datasets/TUBerlin/zeroshot/png_ready_filelist_zero.txt',
    },
}

for ds_name, files in datasets.items():
    print(f"\n{'='*60}")
    print(f"{ds_name}")
    print('='*60)
    
    photo_train = len(open(files['photo_train']).readlines())
    photo_test = len(open(files['photo_test']).readlines())
    sketch_train = len(open(files['sketch_train']).readlines())
    sketch_test = len(open(files['sketch_test']).readlines())
    
    print(f"Photo  - Train: {photo_train:>7}, Test: {photo_test:>7}, Ratio: {photo_train/photo_test:>7.2f}:1")
    print(f"Sketch - Train: {sketch_train:>7}, Test: {sketch_test:>7}, Ratio: {sketch_train/sketch_test:>7.2f}:1")
