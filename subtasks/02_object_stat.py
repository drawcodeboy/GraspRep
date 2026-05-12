import os, sys
sys.path.append(os.getcwd())

BASE_PATH_1 = f"DexGraspNet/data/dexgraspnet" # 5355
BASE_PATH_2 = f"DexGraspNet/data/meshdata" # 5751

def main():
    base_1_dir = os.listdir(BASE_PATH_1)
    base_2_dir = os.listdir(BASE_PATH_2)

    print(len(base_1_dir)) # 5355
    print(len(base_2_dir)) # 5751

    result = 0

    for fname in base_1_dir:
        fname = fname[:-4]
        
        if fname not in base_2_dir:
            result += 1

    print(result)

if __name__ == '__main__':
    main()