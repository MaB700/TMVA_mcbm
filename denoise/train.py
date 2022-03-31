import ROOT

# Setup TMVA
ROOT.TMVA.Tools.Instance()
ROOT.TMVA.PyMethodBase.PyInitialize()
#ROOT.EnableImplicitMT(4)

output = ROOT.TFile.Open("TMVA.root", "RECREATE")
factory = ROOT.TMVA.Factory(
    "mCBM", output,
    "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug")

# Load data
dataloader = ROOT.TMVA.DataLoader("dataset")

data = ROOT.TFile.Open("../mcbm_sim.root")

dataloader.AddRegressionTree(data.Get("train"), 1.0)

for i in range(72 * 32 * 1):
    dataloader.AddVariable("in[{}]".format(i), "in_{}".format(i), "", "F")

for i in range(72 * 32 * 1):
    dataloader.AddTarget("tar[{}]".format(i), "tar_{}".format(i), "")

dataloader.PrepareTrainingAndTestTree(
    ROOT.TCut(""), "SplitMode=Block:NormMode=None:V:!Correlations:!CalcCorrelations:nTrain_regression=24000")
    #:TrainTestSplit_Regression=1.
# Define model
batchLayoutString = "BatchLayout=50|1|2304:"
inputLayoutString = "InputLayout=1|72|32:"
layoutString = "Layout=CONV|16|5|5|1|1|2|2|RELU,CONV|32|5|5|1|1|2|2|RELU,CONV|32|5|5|1|1|2|2|RELU,CONV|16|5|5|1|1|2|2|RELU,CONV|1|3|3|1|1|1|1|TANH,RESHAPE|FLAT:" #,CONV|128|3|3|1|1|1|1|RELU,CONV|128|3|3|1|1|1|1|RELU
trainingString = "TrainingStrategy=MaxEpochs=200,BatchSize=50,ConvergenceSteps=5,Repetitions=1,TestRepetitions=1,Optimizer=ADAM,LearningRate=1e-3:"
cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug:Architecture=GPU"
options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions

# Book methods
factory.BookMethod(dataloader, ROOT.TMVA.Types.kDL, "tmvaDL",
                   options)


method = factory.GetMethod(dataloader.GetName() ,"tmvaDL")
method.Train()
method.WriteStateToFile()

#TODO: add test data evaluation


