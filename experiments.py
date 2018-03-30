def get_experiment(exp_name):
  
    if exp_name == "pascal":
      configList = ["fcn8", "pspnet"]
      datasetList = ["Pascal2012"]
      metricList = ["mIoU"]

      epochs = 1000


    return {"configList":configList, "datasetList":datasetList,
            "metricList":metricList, "epochs":epochs}

