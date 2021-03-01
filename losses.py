import tensorflow as tf
import tensorflow.keras as tfk

class MultiClassDice(tfk.losses.Loss):

    def __init__(self):

        super().__init__()
        self.smooth = 1e-5

    def call(self, y_true, y_pred):
        

        y_pred = tfk.backend.softmax(y_pred)
        
        tsum = tfk.backend.reshape(y_true[...,1:], (1, -1))
        psum = tfk.backend.reshape(y_pred[...,1:], (1, -1))

        intersection = psum * tsum
        sums = psum + tsum

        intersection = tfk.backend.sum(intersection) + self.smooth
        sums = tfk.backend.sum(sums) + self.smooth

        score = 2.0 * intersection / sums

        return (1 - tfk.backend.mean(score))



class MultiClassDiceXent(tfk.losses.Loss):

    def __init__(self):

        super().__init__()
        self.smooth = 1e-5

    def call(self, y_true, y_pred):

        y_pred = tfk.backend.softmax(y_pred)

        y_true_xe = tfk.backend.cast(y_true, y_pred.dtype)
        xent = tfk.backend.categorical_crossentropy(y_true_xe, y_pred)

        tsum = tfk.backend.reshape(y_true[...,1:], (1, -1))
        psum = tfk.backend.reshape(y_pred[...,1:], (1, -1))
        
        intersection = psum * tsum
        sums = psum + tsum

        intersection = tfk.backend.sum(intersection) + self.smooth
        sums = tfk.backend.sum(sums) + self.smooth

        score = 2.0 * intersection / sums

        return (1 - tfk.backend.mean(score)) + xent