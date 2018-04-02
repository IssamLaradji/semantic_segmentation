def get_experiment(exp_name):
  
    if exp_name == "pascal":
      configList = ["resfcn", "segnet", "gcn","pspnet", "fcn8"]
      #configList = ["gcn", "segnet","pspnet", "fcn8"]
      datasetList = ["Pascal2012"]
      metricList = ["mIoU"]

      epochs = 1000


    return {"configList":configList, "datasetList":datasetList,
            "metricList":metricList, "epochs":epochs}

