#include <string>
#include <iostream>
#include <vector>
#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA/MethodDL.h"

int predict(){

    TMVA::Reader* reader = new TMVA::Reader("!Silent");
    TFile* f = new TFile("../mcbm_sim.root");
    TTree* t = (TTree*)f->Get("train");
    Float_t in[2304];
    Float_t tar[2304];
    t->SetBranchAddress("in", in);
    t->SetBranchAddress("tar", tar);
    t->GetEntry(21000); //do single pred on some test event
    
    for(int i{}; i < 2304; i++){
        std::string varName = "in[" + std::to_string(i) + "]"; 
        reader->AddVariable(varName, &in[i]);
    }
    
    TMVA::IMethod* method = reader->BookMVA("DL", "dataset/weights/mCBM_tmvaDL.weights.xml");
    // --- optional: set output function Softmax -> ArgMax or only ArgMax
    // TMVA::MethodDL* mDL = dynamic_cast<TMVA::MethodDL*>(method);
    // mDL->SetOutputFunction(TMVA::DNN::EOutputFunction::kSoftmax); 
    
    std::vector<Float_t> out = reader->EvaluateRegression("DL");
    std::cout << "pred    " << "   true" << std::endl;
    for(int i{}; i < 2304; i++){
        if(tar[i] < 0.9) continue;
        std::cout << out[i] << "   " << tar[i] << std::endl;
    }
    return 0;
}