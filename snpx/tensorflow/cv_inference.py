import os
import cv2
import mv_dataset as data


class OpenCVInference(object):
    """ 
    """
    def __init__(self, pb_model, input_size):
        self.net = cv2.dnn.readNetFromTensorflow(pb_model)
        self.shape = (input_size, input_size)

    def __call__(self, image):
        """ """
        image = cv2.resize(image, self.shape)
        image = cv2.dnn.blobFromImage(image, (0.00392), self.shape)
        self.net.setInput(image)
        out = self.net.forward()
        print (out.shape)
        return out

def main():
    """ """
    images = [21158, 35728, 37021, 76090, 79764]

    inference = OpenCVInference('mobilenet.pb', 128)
    for i in images:
        url = os.path.join(data.DATASET_DIR, 'train_patch', 'training_'+str(i)+'.jpg')
        image = cv2.imread(url)
        inference(image)
        print ('URL: training_', i, '(', h, w, ')')

if __name__ == '__main__':
    main()