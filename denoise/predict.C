#include <string>
#include <iostream>
#include <vector>
#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA/MethodDL.h"
#include "TStopwatch.h"

int predict(){
    ROOT::EnableImplicitMT(0);
    TMVA::Reader* reader = new TMVA::Reader("!Silent");
    TFile* f = new TFile("../mcbm_sim.root");
    TTree* t = (TTree*)f->Get("train");
    Float_t in[2304];
    Float_t tar[2304];
    t->SetBranchAddress("in", in);
    t->SetBranchAddress("tar", tar);
    //t->GetEntry(21000); //do single pred on some test event
    
    for(int i{}; i < 2304; i++){
        std::string varName = "in[" + std::to_string(i) + "]"; 
        reader->AddVariable(varName, &in[i]);
    }
    TStopwatch timer;
    timer.Start();
    TMVA::IMethod* method = reader->BookMVA("DL", "./dataset/weights/mCBM_tmvaDL.weights.xml");// takes 0.2s
    timer.Stop(); //dataset/weights/mCBM_tmvaDL.weights.xml
    Double_t dt1 = timer.RealTime();
    std::cout << "Booking Method took " << dt1 << "s" << std::endl; 

    // --- optional: set output function Softmax -> ArgMax or only ArgMax
    // TMVA::MethodDL* mDL = dynamic_cast<TMVA::MethodDL*>(method);
    // mDL->SetOutputFunction(TMVA::DNN::EOutputFunction::kSoftmax); 
    std::vector<Float_t> out;
    Float_t allHitAvgDiff{};
    Float_t trueHitAvg{};
    Float_t noiseHitAvg{};
    Float_t allPixelAvgDiff{};

    int nofEvents{};

    for(int i{20000}; i < 20000+10; i++){ //t->GetEntriesFast()
        nofEvents++;
        t->GetEntry(i);
        timer.Start();
        out = reader->EvaluateRegression("DL"); // takes 0.15s :(
        timer.Stop();
        Double_t dt2 = timer.RealTime();
        std::cout << "Event: " << i << "  Time to eval Regress: " << dt2 << "s"<< std::endl;
        int allHits{};
        Float_t allHitsSumDiff{};
        int trueHits{};
        Float_t trueHitsSum{};
        int noiseHits{};
        Float_t noiseHitsSum{};
        Float_t allPixelSumDiff{};

        for(int j{}; j < 2304; j++){
            allPixelSumDiff += TMath::Abs(tar[j] - out[j]);
            if( (in[j] > 0.9) && (tar[j] > 0.9) ){ // true hit
                trueHits++;
                trueHitsSum += out[j];
            }
            if( (in[j] > 0.9) && (tar[j] < 0.1) ){ // noise hit
                noiseHits++;
                noiseHitsSum += out[j];
            }
            if( in[j] > 0.9 ){ // diff tar&pred for true hit + noise hit
                allHits++;
                allHitsSumDiff += TMath::Abs(tar[j] - out[j]);
            }
        }
        std::cout << "in tar pred" << std::endl;
        for(int j{}; j < 2304; j++){
            if(in[j] > 0.9){
                std::cout << in[j] << "  " << tar[j] << "  " << out[j] << std::endl;
            }
        }

        allHitAvgDiff += allHitsSumDiff / allHits;
        trueHitAvg += trueHitsSum / trueHits;
        noiseHitAvg += noiseHitsSum / noiseHits;
        allPixelAvgDiff += allPixelSumDiff / 2304;
    }

    std::cout << "HitAverageDiff : " << allHitAvgDiff / nofEvents << std::endl; // -> 0
    std::cout << "trueHitAverage : " << trueHitAvg / nofEvents << std::endl; // -> 1
    std::cout << "noiseHitAverage : " << noiseHitAvg / nofEvents << std::endl; // -> 0
    std::cout << "allPixelAverageDiff : " << allPixelAvgDiff / nofEvents << std::endl; // -> 
    
    /* std::cout << "true" << "      pred" << std::endl;
    for(int i{}; i < 2304; i++){
        if(tar[i] < 0.9) continue;
        std::cout << tar[i] << "   " << out[i] << std::endl;
    } */
    return 0;
}

