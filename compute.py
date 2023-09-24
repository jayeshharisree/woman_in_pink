import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


def compute(a, b, c, d, e, z, g, h, i):
    # Preparing the data:
    data_file_name = 'breast-cancer-wisconsin.data.txt'

    first_line = "id,clump_thickness,unif_cell_size,unif_cell_shape,marg_adhesion,single_epith_cell_size,bare_nuclei,bland_chrom,norm_nucleoli,mitoses,class"
    with open(data_file_name, "r+") as f:
        content = f.read()
        f.seek(0, 0)
        f.write(first_line.rstrip('\r\n') + '\n' + content)
    df = pd.read_csv(data_file_name)
    df.replace('?', np.nan, inplace=True)
    df.dropna(inplace=True)
    df.drop(['id'], axis=1, inplace=True)

    df['class'].replace('2', 0, inplace=True)
    df['class'].replace('4', 1, inplace=True)

    df.to_csv("combined_data.csv", index=False)

    # Data sets
    CANCER_TRAINING = "cancer_training.csv"
    CANCER_TEST = "cancer_test.csv"

    # Load datasets.
    training_data = pd.read_csv(CANCER_TRAINING)
    X_training = training_data.iloc[:, :-1]
    y_training = training_data.iloc[:, [-1]]
    classifier = KNeighborsClassifier(n_neighbors=7)

    # Fit model.
    classifier = classifier.fit(X_training, y_training)
    k = a
    l = b
    m = c
    n = d
    o = e
    p = z
    q = g
    r = h
    s = i

    def new_samples():
        return np.array([[k, l, m, n, o, p, q, r, s],
                         ], dtype=np.float32)

    samp = new_samples()
    s = list(classifier.predict(samp))
    print(s)

    if (s == [[1]]):
        return "  Malignant (Cancerous) ! \n Please, Take Immediate Action & Consult a Doctor "
    else:
        return "  Benign (Non-Cancerous) ! \n It's not Dangerous but Consultation is Recommended "


if __name__ == '__main__':
    print(compute(a, b, c, d, e, f, g, h, i))
