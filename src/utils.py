import sys
import os 

def retag(feature_file_path, tag_file_path, last_feature = False):
    """
    Re-tag the feature file with the tags in the tag file.
    Both files are CRFPP format.
    """
    f_f = open(feature_file_path, "r")
    f_t = open(tag_file_path, "r")
    lines_f = f_f.readlines()
    lines_t = f_t.readlines()
    f_f.close()
    f_t.close()
    assert len(lines_f) == len(lines_t)

    with open(feature_file_path, "w") as f:
        for (l_f, l_t) in zip(lines_f, lines_t):
            if len(l_f) <= 5:#empty
                f.write("\n")
            else:
                if last_feature is False:
                    features = l_f.strip().split()[:-1]
                else:
                    features = l_f.strip().split()[:]
                tag = l_t.strip().split()[-1]
            f.write("\t".join(features) + "\t" + tag + "\n")      


