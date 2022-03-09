import ROOT

# Setup TMVA
ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()
#ROOT.EnableImplicitMT(4)

output = ROOT.TFile.Open("TMVA.root", "RECREATE")
factory = ROOT.TMVA.Factory(
    "MNIST", output,
    "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug")

# Load data
dataloader = ROOT.TMVA.DataLoader("dataset")

data = ROOT.TFile.Open("mnist0.root")

dataloader.AddRegressionTree(data.Get("train"), 1.0)

for i in range(28 * 28 * 1):
    dataloader.AddVariable("x[{}]".format(i), "x_{}".format(i), "", "F", 0.0, 1.0)

for i in range(10):
    dataloader.AddTarget("y[{}]".format(i), "y_{}".format(i), "", 0.0, 1.0)

dataloader.PrepareTrainingAndTestTree(
    ROOT.TCut(""), "SplitMode=Random:NormMode=None:V:!Correlations:!CalcCorrelations")

# Define model
batchLayoutString = "BatchLayout=100|1|784:"
inputLayoutString = "InputLayout=1|28|28:"
layoutString = "Layout=CONV|4|3|3|1|1|1|1|RELU,MAXPOOL|2|2|1|1,RESHAPE|FLAT,DENSE|16|RELU,DENSE|10|SIGMOID:"
trainingString = "TrainingStrategy=MaxEpochs=100,BatchSize=100,Optimizer=ADAM,LearningRate=1e-3:"
cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug:Architecture=GPU"
options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions

# Book methods
factory.BookMethod(dataloader, ROOT.TMVA.Types.kDL, "tmvaDL",
                   options)

# Set output function -- TODO: using softmax yield worst results, may use different ErrorStrategy
# only use output function for prediction ?
# factory.GetMethod(dataloader.GetName() ,"tmvaDL").SetOutputFunction(ROOT.TMVA.DNN.EOutputFunction.kSoftmax)

# Run training, test and evaluation
factory.TrainAllMethods()
#factory.TestAllMethods()
#factory.EvaluateAllMethods()

