from sklearn.metrics import confusion_matrix
import utils_misc as ut 
import numpy as np


class mIoU:
    def __init__(self):
        self.cf = None

    def _average(self, cf):
        Inter = np.diag(cf)
        G = cf.sum(axis=1)
        P = cf.sum(axis=0)
        union =  G + P - Inter

        nz = union != 0
        mIoU = Inter[nz] / union[nz]
        mIoU = np.mean(mIoU)

        return mIoU

    def scoreBatch(self, model, batch):    
        model.eval()

        preds = ut.t2n(model.predict(batch, metric="labels"))
        labels = ut.t2n(batch["labels"])

        cf = confusion_matrix(y_true=preds.ravel(),
                         y_pred=labels.ravel(),
                         labels=np.arange(model.n_classes))

        return {"score": self._average(cf), "cf":cf}


    def update_running_average(self, model, batch):
        cf = self.scoreBatch(model, batch)["cf"]
        if self.cf is None:
            self.cf = cf
        else:
            self.cf += cf

    def get_running_average(self):
        return self._average(self.cf) 

