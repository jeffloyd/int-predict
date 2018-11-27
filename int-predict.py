"""
================================
Raspberry Pi Cam Predicting hand-written digits
================================
"""
print(__doc__)

# Python imports
import matplotlib.pyplot as plot
from sklearn import datasets, svm, metrics
digits = datasets.load_digits()

images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plot.subplot(2, 4, index + 1)
    plot.axis('off')
    plot.imshow(image, cmap=plot.cm.gray_r, interpolation='nearest')
    plot.title('Train: %i' % label)

# flatten the data into (sampledata, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a support vector classifier
classit = svm.SVC(gamma=0.001)

# training on half the dataset
classit.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# predict the remaining half:
expected = digits.target[n_samples // 2:]
predicted = classit.predict(data[n_samples // 2:])

print("Classification report %s:\n%s\n"
      % (classit, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plot.subplot(2, 4, index + 5)
    plot.axis('off')
    plot.imshow(image, cmap=plot.cm.gray_r, interpolation='nearest')
    plot.title('Prediction: %i' % prediction)

#show the results
plot.show()
