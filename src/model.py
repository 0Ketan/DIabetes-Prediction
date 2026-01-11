from sklearn import svm


def create_model():
    classifier = svm.SVC(kernel='linear')
    return classifier