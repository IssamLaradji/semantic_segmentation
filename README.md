1. Debugging
------------
You can debug the code by running,
`python main.py -e fcn8_pascal -m debug -d PascalPoints`

which lets you interact with the model and dataset using the python debugger.

- `PascalPoints` is a dataset class defined in the datasets folder.

- `fcn8_pascal` is an experiment description defined in `experiments.py`

You can test the model as follows,

```python
 # Load the model and optimizer
 model, opt, _ = mu.init_model_and_opt(main_dict)


 # Load the training and val set
 train_set, val_set = mu.load_trainval(main_dict)

 # Get a batch from the training set at index 15
 batch = ut.get_batch(train_set, indices=[15])

 # Use the model to get the segmentaiton and probaility outputs
 model.predict(batch, "blobs")
 model.predict(batch, "probs")

 # Obtain the score
 score = val.valBatch(model, batch, metric_name="mIoU")

 # Train the model on the batch
 tr.fitBatch(model, batch, opt=opt, loss_name="wtp_loss", epochs=50)

 # Obtain the new score
 score = val.valBatch(model, batch, metric_name="mIoU")
```

2. Testing
----------
You can evaluate your model by running the following command,
`python main.py -e fcn8_pascal -m test -d PascalPoints`

The command above should give you an output like this,

```
semantic_segmentation: python main.py -e fcn8_pascal -m test -d PascalPoints 
CUDA: 9.1.85
Pytroch: 0.4.0
PascalPoints
PascalPoints - fcn8 - wtp_loss
loaded best model...
Validating... 736
0 - 0/736 - Validating test set - mIoU: 0.412
0 - 73/736 - Validating test set - mIoU: 0.035
0 - 146/736 - Validating test set - mIoU: 0.035
```

3. Training
-----------
You can train your model by running the command:

`python main.py -e fcn8_pascal -m train -d PascalPoints -r reset`

which should give you an output that looks like this,

```
 'testTransformer': 'Te_WTP',
 'trainTransformer': 'Tr_WTP',
 'val_batchsize': 1,
 'verbose': 1}

EXP: fcn8_dataset:PascalPoints_metric:mIoU_loss:wtp_loss,  Reset: reset
-----------------------------------------------------------------------
        dataset  n_train  n_val
0  PascalPoints     8498    736
TRAINING FROM SCRATCH EPOCH: 0/1000
Training Epoch 1 .... 8498 batches
1 - (0/8498) - train - mIoU: 0.078 - wtp_loss: 10.061 - elapsed: 0.008

```





