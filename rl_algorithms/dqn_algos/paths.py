from rl_utils.env_wrapper.blob_env import get_path_traveled

min_path_len=1000

def check_min_path(path_len):
    if path_len<min_path_len:
        min_path_len=path_len
        path_travel=get_path_traveled()
        with open('output.txt', 'w') as f:
            # Write the min_path as a string
            f.write("Minimum Path:\n")
            f.write(", ".join(map(str, path_travel)) + "\n")  # Join list items as a string
            # Write the min_pathlen
            f.write("Minimum Path Length: {}\n".format(min_path_len))    

    