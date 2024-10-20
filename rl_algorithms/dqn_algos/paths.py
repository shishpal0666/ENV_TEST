from rl_utils.env_wrapper.blob_env import get_path_traveled

class path_saver:
  def __init__(self):
    self.min_path_len = 1000

  def check_min_path(self, path_len, ep_no, pathfilename):
    if path_len < self.min_path_len:
        self.min_path_len = path_len
        path_travel = get_path_traveled()
        with open(pathfilename, 'w') as f:
            # Write the min_path as a string
            f.write("Minimum Path:\n")
            f.write(", ".join(map(str, path_travel)) + "\n")  # Join list items as a string
            # Write the min_pathlen
            f.write("Minimum Path Length: {}\n".format(self.min_path_len))
            f.write("Episode no: {}\n".format(ep_no))