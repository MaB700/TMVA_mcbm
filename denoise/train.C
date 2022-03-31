#include <iostream>
#include <string>

#if !defined(__CLING__)
#include "TFile.h"
#include "TTree.h"
#include "TSystem.h"
#include "TMVA/Tools.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Factory.h"
#include "TMVA/MethodDL.h"
#include "TCanvas.h"
#endif

//R__LOAD_LIBRARY(predict_batch_C)
void train(){

    TMVA::Tools::Instance();
    /* TFile* output = TFile::Open("TMVA.root", "RECREATE");
    TMVA::Factory* factory = new TMVA::Factory(
        "mCBM", output,
        "V:!Silent:Color:DrawProgressBar:!ROC:ModelPersistence:AnalysisType=Regression:Transformations=None:!Correlations:VerboseLevel=Debug"
    );

    // load data
    TMVA::DataLoader* dataloader = new TMVA::DataLoader("dataset");
    TFile* data = TFile::Open("../mcbm_sim.root");
    //dataloader->AddRegressionTarget(); //FIXME:
    dataloader->AddRegressionTree((TTree*)data->Get("train"), 1.0);     

    for(int i{} ; i < 72*32 ; i++){
        dataloader->AddVariable("in[" + std::to_string(i) + "]", "in_" + std::to_string(i), "" );
    }
    // or AddVariableArray
    for(int i{} ; i < 72*32 ; i++){
        dataloader->AddTarget("tar[" + std::to_string(i) + "]", "tar_" + std::to_string(i), "" );
    }

    dataloader->PrepareTrainingAndTestTree(TCut(""), "SplitMode=Block:NormMode=None:V:!Correlations:!CalcCorrelations:nTrain_regression=24000");

    // define model & options
    TString batchLayoutString = "BatchLayout=50|1|2304:";
    TString inputLayoutString = "InputLayout=1|72|32:";
    TString layoutString = "Layout=CONV|16|5|5|1|1|2|2|RELU,CONV|32|5|5|1|1|2|2|RELU,CONV|32|5|5|1|1|2|2|RELU,CONV|16|5|5|1|1|2|2|RELU,CONV|1|3|3|1|1|1|1|TANH,RESHAPE|FLAT:";
    TString trainingString = "TrainingStrategy=MaxEpochs=20,BatchSize=50,ConvergenceSteps=5,Repetitions=1,TestRepetitions=1,Optimizer=ADAM,LearningRate=1e-3:";
    TString cnnOptions = "H:V:VarTransform=None:ErrorStrategy=SUMOFSQUARES:VerbosityLevel=Debug:Architecture=GPU";
    TString options = batchLayoutString + inputLayoutString + layoutString + trainingString + cnnOptions;
    // Book method
    factory->BookMethod(dataloader, TMVA::Types::kDL, "tmvaDL", options);
    TMVA::MethodDL* method = dynamic_cast<TMVA::MethodDL*>(factory->GetMethod(dataloader->GetName(), "tmvaDL"));
    method->Train();
    method->WriteStateToFile(); */
    //gROOT->ProcessLine("gSystem->AddIncludePath(\"-I/usr/local/cuda-11.4/targets/x86_64-linux/include\");.L predict_batch.C+;predict_batch();"); //TODO: does not work
    
    gROOT->ProcessLine("gSystem->Load(\"predict_batch_C.so\");predict_batch();");


}
