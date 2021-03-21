import numpy as np


def load_1d_arr(filename, datatype):
    with open(filename) as f:
        lines = "".join(f.readlines())[1:-2].replace(datatype,"")
    return np.fromstring(lines, sep=",")



def load_2d_arr(filename, datatype):
    f = open(filename, 'r')
    lines = "".join(f.readlines())   \
              .replace(datatype,"")  \
              .replace(".nan","nan") \
              .replace("\n","")      \
              .replace("],","\n")    \
              .replace("[","")       \
              .replace("]","")       \
              .replace(",","")
    f.close()

    new_filename = filename + ".tmp"
    f = open(new_filename, 'w+')
    lines = f.write(lines)
    f.close()

    cont = np.loadtxt(new_filename, ndmin=2)
    cont[np.isnan(cont)] = -32768
    cont = cont.astype(int)
    return cont


if __name__ == "__main__":
    means = load_1d_arr("./peru.out.means.txt", "f32")
    breaks = load_1d_arr("./peru.out.breaks.txt", "i32")
    np.save("./peru.out.means.npy", means)
    np.save("./peru.out.breaks.npy", breaks)

    means = load_1d_arr("./sahara.out.means.txt", "f32")
    breaks = load_1d_arr("./sahara.out.breaks.txt", "i32")
    np.save("./sahara.out.means.npy", means)
    np.save("./sahara.out.breaks.npy", breaks)

    peru = load_2d_arr("./peru.in.images.txt", "f32")
    sahara = load_2d_arr("./sahara.in.images.txt", "f32")
    np.save("./peru.in.npy", peru)
    np.save("./sahara.in.npy", sahara)
