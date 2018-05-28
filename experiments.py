
def get_experiment(exp_name):

		configList = None
		datasetList = None
		metricList = None
		epochs= 1000
		lossList= None

		if exp_name == "fcn8_pascal":
			configList = ["fcn8"]
			datasetList = [ "PascalPoints"]
			metricList = ["mIoU"]
			lossList = ["wtp_loss"]
			epochs = 1000


		return {"configList":configList, 
				"datasetList":datasetList,
				"metricList":metricList, 
				"epochs":epochs,
				"lossList":lossList}

