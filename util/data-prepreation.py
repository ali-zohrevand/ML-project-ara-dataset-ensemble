from merg_files import merge_files
for directory in range(2):
    print(f"directory {directory+1}")
    merge_files(f'House{directory + 1}')


