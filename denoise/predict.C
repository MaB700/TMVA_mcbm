#include <string>
#include <iostream>
#include <vector>
#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA/MethodDL.h"

int predict(){

    TMVA::Reader* reader = new TMVA::Reader("!Silent");
    TFile* f = new TFile("mnist0.root");
    TTree* t = (TTree*)f->Get("test");
    Float_t x[784];
    Float_t y[10];
    t->SetBranchAddress("x", x);
    t->SetBranchAddress("y", y);
    t->GetEntry(0); //do single pred on first event
    
    for(int i{}; i < 784; i++){
        std::string varName = "x[" + std::to_string(i) + "]"; 
        reader->AddVariable(varName, &x[i]);
    }
    
    TMVA::IMethod* method = reader->BookMVA("DL", "dataset/weights/MNIST_tmvaDL.weights.xml");
    // --- optional: set output function Softmax -> ArgMax or only ArgMax
    // TMVA::MethodDL* mDL = dynamic_cast<TMVA::MethodDL*>(method);
    // mDL->SetOutputFunction(TMVA::DNN::EOutputFunction::kSoftmax); 
    
    std::vector<Float_t> out = reader->EvaluateRegression("DL");
    std::cout << "pred    " << "   true" << std::endl;
    for(int i{}; i < 10; i++){
        std::cout << out[i] << "   " << y[i] << std::endl;
    }
    return 0;
}