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

def strip(file_name, kept_list = ["author", "title"], inplace = True):
    """
    Repalce the tags with None excpet those in the kept_list
    """
    lines = open(file_name, "r").readlines()
    with open(file_name, "w") as f:
        for line in lines:
            if len(line) <= 5:#empty
                f.write("\n")
            else:
                features = line.strip().split()[:-1]
                tag = line.strip().split()[-1]
                if tag not in kept_list:
                    tag = "None"
                f.write("\t".join(features) + "\t" + tag + "\n")


def conditional_join(file_one, file_two, output_file):
    """
    Only keep the tokens with the same prediction in both file_one and file_two
    """
    lines_1 = open(file_one, "r").readlines()
    lines_2 = open(file_two, "r").readlines()
    assert len(lines_1) == len(lines_2)
    with open(output_file, "w") as f:
        for (l_1, l_2) in zip(lines_1, lines_2):
            if len(l_1) <= 5:
                f.write("\n")
            elif l_1.strip().split()[-1] == l_2.strip().split()[-1]:
                f.write(l_1)
    

