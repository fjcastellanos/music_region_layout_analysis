# parse_data_end_to_end.py
import sys

if __name__ == "__main__":
    in_file  = sys.argv[1]
    in2_file = sys.argv[2]
    out_file = sys.argv[3]

    f = open(in_file,"r")
    lines_in = f.readlines()
    f2 = open(in2_file,"r")
    lines_in2 = f2.readlines()

    lines_in.sort()
    lines_in2.sort()

    lines_write = []

    for line1, line2 in zip(lines_in, lines_in2):
        line2 = line2.replace('datasets/JSON','data/SRC')[:-6]
        # line1 = line1.replace('outputs_json','data/outputs_json')[:-1]
        line1 = line1.replace('end_to_end/','')[:-1]
        lines_write.append(line2 + "\t" + line1 + "\n")
        # print(line2 + " " + line1[:-1])

    f_out = open(out_file, "w")
    # print(f'Lines to write: {lines_write}')
    f_out.writelines(lines_write)
