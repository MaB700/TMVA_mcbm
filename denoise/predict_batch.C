#include <string>
#include <iostream>
#include <vector>


#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA/MethodDL.h"
#include "TStopwatch.h"
#include "TMVA/ROCCurve.h"
#include "TGraph.h"
#ifdef R__HAS_CUDA
#include "TMVA/DNN/Architectures/Cuda.h"
#endif



using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

using Architecture_t = TMVA::DNN::TCuda<Float_t>;
using Scalar_t = typename Architecture_t::Scalar_t;
using Matrix_t = typename Architecture_t::Matrix_t;
using Tensor_t = typename Architecture_t::Tensor_t;
using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
using TMVAInput_t =  std::tuple<const std::vector<TMVA::Event *> &, const TMVA::DataSetInfo &>;
using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

DeepNet_t* ReadModelFromXML(TString, size_t);
std::vector<TMVA::Event *> loadEvents(Long64_t, Long64_t,  const char*);

int predict_batch(  Float_t NNthreshold = 0.8,
                    Long64_t nEvents = 1000, 
                    const char* inFile = "../mcbm_sim.root", 
                    size_t batchSize = 50)
{
    
    TMVA::Tools::Instance();
    Architecture_t::SetRandomSeed(10);    
    
    const std::vector<TMVA::Event *> allData = loadEvents(0, nEvents, inFile);
      
    TString fileXML = "dataset/weights/mCBM_tmvaDL.weights.xml";
    DeepNet_t* fNet = ReadModelFromXML(fileXML, batchSize);

    // Loading the test dataset
    TMVA::DataSetInfo dsix;
    for(int i{0}; i < 72*32; i++){
        TString in1; in1.Form("in[%d]", i);
        TString in2; in2.Form("in_%d", i);
        dsix.AddVariable(in1, in2, "");

        TString out1; out1.Form("tar[%d]", i);
        TString out2; out2.Form("tar_%d", i);
        dsix.AddTarget(out1, out2, "", (0.0), (0.0));
    }
    TMVAInput_t testTuple = std::tie(allData, dsix); 
    TensorDataLoader_t testData(testTuple, allData.size(), fNet->GetBatchSize(),
                                    {fNet->GetInputDepth(), fNet->GetInputHeight(), fNet->GetInputWidth()},
                                    {fNet->GetBatchDepth(), fNet->GetBatchHeight(), fNet->GetBatchWidth()} ,
                                    fNet->GetOutputWidth(), 1);
    
    TFile* outFile = new TFile("mrich_nn_eval.root", "RECREATE");
    // Histograms
    TH1F* allHitResponse = new TH1F("allHitResponse", "NN response for all hits", 50, 0., 1.);
    allHitResponse->GetXaxis()->SetTitle("NN response"); allHitResponse->GetYaxis()->SetTitle("count");    
    TH1F* trueHitResponse = new TH1F("trueHitResponse", "NN response for true hits", 50, 0., 1.);
    trueHitResponse->GetXaxis()->SetTitle("NN response"); trueHitResponse->GetYaxis()->SetTitle("count");
    TH1F* noiseHitResponse = new TH1F("noiseHitResponse", "NN response for noise hits", 50, 0., 1.);
    noiseHitResponse->GetXaxis()->SetTitle("NN response"); noiseHitResponse->GetYaxis()->SetTitle("count");

    TH1F* trueHitResponseTH = new TH1F("trueHitResponseTH", "123", 10, -4., 5.);
    TH1F* noiseHitResponseTH = new TH1F("noiseHitResponseTH", "1234", 10, -4., 5.);
    // both hists together build the confusion matrix

    std::vector<Bool_t> tarx;
    std::vector<Float_t> predx;
    std::vector<Bool_t> tarn;
    std::vector<Float_t> predn;
    std::vector<Float_t> weightsn;

    TStopwatch timer;
    timer.Start();
    for (auto batch : testData) {
        auto inputTensor = batch.GetInput();
        TMatrixT<Float_t> inMat(inputTensor.GetMatrix()); // (1) 2304 50            
        auto targetMatrix = batch.GetOutput();
        TMatrixT<Float_t> tarMat(targetMatrix); // 50 2304            
        
        fNet->Forward(inputTensor);
        
        auto pred = fNet->GetLayers().back()->GetOutputAt(0);
        TMatrixT<Float_t> predMat(pred); // 50 2304
            
        for(size_t i{}; i < fNet->GetBatchSize(); i++){
            for(size_t j{}; j < 2304; j++){
                if(inMat(j, i) > 0.9 && tarMat(i, j) > 0.9){ // true hit
                    trueHitResponse->Fill(predMat(i, j));
                    if(predMat(i, j) > NNthreshold){
                        trueHitResponseTH->Fill(3.); // right after applying threshold
                        tarx.push_back(1);
                    }else{
                        trueHitResponseTH->Fill(-2.); // wrong after applying threshold
                        tarx.push_back(0);
                    }
                    predx.push_back(predMat(i, j));
                }
                else if(inMat(j, i) > 0.9 && tarMat(i, j) < 0.1){ // noise hit
                    noiseHitResponse->Fill(predMat(i, j));
                    if(predMat(i, j) > NNthreshold){
                        noiseHitResponseTH->Fill(-2.); // wrong after applying threshold
                        tarn.push_back(0);
                        weightsn.push_back(1.);
                    }else{
                        noiseHitResponseTH->Fill(3.); // right after applying threshold
                        tarn.push_back(1);
                        weightsn.push_back(1.);
                    }
                    predn.push_back(predMat(i, j));
                }
                if(inMat(j, i) > 0.9) allHitResponse->Fill(predMat(i, j));             
            }
        }
    }   
    
    timer.Stop();
    Double_t dt1 = timer.RealTime();
    std::cout << "Time to eval "<<nEvents<<" events on gpu with batchsize " << batchSize << std::endl;
    std::cout << "time: " << dt1 << "s" << std::endl;
    TMVA::ROCCurve* rocx = new TMVA::ROCCurve(predx, tarx);
    TMVA::ROCCurve* roc = new TMVA::ROCCurve(predn, tarn, weightsn);
    TGraph* gRocx = rocx->GetROCCurve();
    TGraph* gRoc = roc->GetROCCurve();
    std::cout << "AUC true  : " << rocx->GetROCIntegral() << std::endl;
    std::cout << "AUC noise : " << roc->GetROCIntegral() << std::endl;
    outFile->cd();
    gRocx->Write("RocTrueHits");
    gRoc->Write("RocNoiseHits");
    allHitResponse->Write();
    trueHitResponse->Write();
    noiseHitResponse->Write();
    trueHitResponseTH->Write();
    noiseHitResponseTH->Write();
    //outFile->Write(); //write all hists at once
    outFile->Close();

    return 1;
}

std::vector<TMVA::Event *> loadEvents(Long64_t firstEvt, Long64_t lastEvt,  const char* iFile){ 

    TStopwatch timer;
    timer.Start();
    std::cout << "loading Events from file ... \n" ; 
    TFile *file = TFile::Open(iFile); 
    TTree *tree = (TTree*)file->Get("train"); 
    Float_t in[2304];
    Float_t tar[2304];
    tree->SetBranchAddress("in", in);
    tree->SetBranchAddress("tar", tar);
    std::vector<TMVA::Event*> allData;
        
    Long64_t nofEntries = tree->GetEntries();
    if(firstEvt > nofEntries) firstEvt = 0;
    if(lastEvt > nofEntries || lastEvt == 0) lastEvt = nofEntries;
    for(Long64_t i= firstEvt; i < lastEvt; i++){
        tree->GetEntry(i);
        std::vector<Float_t> input; 
        std::vector<Float_t> target; 
        
        for(int j=0; j < 2304; j++){
            target.push_back(tar[j]);
            input.push_back(in[j]);
        }
        

        TMVA::Event* ev = new TMVA::Event(input, target);
        allData.push_back(ev); 
    }
    timer.Stop();
    Double_t dt = timer.RealTime();
    std::cout << "loading finished ! \n" ; 
    std::cout << "time for loading Events  " << dt << "s\n";  
    file->Close();
    return allData;
}

DeepNet_t* ReadModelFromXML(TString xmlFile, size_t inBatchSize){

    void* model = TMVA::gTools().xmlengine().ParseFile(xmlFile);
    void* rootnode = TMVA::gTools().xmlengine().DocGetRootElement(model);
    //ReadModelFromXML(rootnode, deepNet);
    auto netXML = TMVA::gTools().GetChild(rootnode, "Weights");
    if (!netXML){
        netXML = rootnode;
    }

    size_t netDepth;
    TMVA::gTools().ReadAttr(netXML, "NetDepth", netDepth);

    size_t inputDepth, inputHeight, inputWidth;
    TMVA::gTools().ReadAttr(netXML, "InputDepth", inputDepth);
    TMVA::gTools().ReadAttr(netXML, "InputHeight", inputHeight);
    TMVA::gTools().ReadAttr(netXML, "InputWidth", inputWidth);
    //std::cout << "inputDepth" << inputDepth << std::endl;
    size_t batchSize, batchDepth, batchHeight, batchWidth;
    TMVA::gTools().ReadAttr(netXML, "BatchSize", batchSize);
    // use always batchsize = 1
    //batchSize = 1;
    TMVA::gTools().ReadAttr(netXML, "BatchDepth", batchDepth);
    TMVA::gTools().ReadAttr(netXML, "BatchHeight", batchHeight);
    TMVA::gTools().ReadAttr(netXML, "BatchWidth",  batchWidth);

    char lossFunctionChar;
    TMVA::gTools().ReadAttr(netXML, "LossFunction", lossFunctionChar);
    char initializationChar;
    TMVA::gTools().ReadAttr(netXML, "Initialization", initializationChar);
    char regularizationChar;
    TMVA::gTools().ReadAttr(netXML, "Regularization", regularizationChar);
    //char outputFunctionChar;
    //TMVA::gTools().ReadAttr(netXML, "OutputFunction", outputFunctionChar);
    double weightDecay;
    TMVA::gTools().ReadAttr(netXML, "WeightDecay", weightDecay);

    // ---- create deepnet ----
    DeepNet_t* fNetx = new DeepNet_t(inBatchSize, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, 
                            static_cast<ELossFunction>(lossFunctionChar),
                            static_cast<EInitialization>(initializationChar),
                            static_cast<ERegularization>(regularizationChar),
                            weightDecay);
    
    auto layerXML = TMVA::gTools().xmlengine().GetChild(netXML);

    // loop on the layer and add them to the network
    for (size_t i = 0; i < netDepth; i++) {

        TString layerName = TMVA::gTools().xmlengine().GetNodeName(layerXML);

        // case of dense layer
        if (layerName == "DenseLayer") {

            // read width and activation function and then we can create the layer
            size_t width = 0;
            TMVA::gTools().ReadAttr(layerXML, "Width", width);

            // Read activation function.
            TString funcString;
            TMVA::gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
            EActivationFunction func = static_cast<EActivationFunction>(funcString.Atoi());


            fNetx->AddDenseLayer(width, func, 0.0); // no need to pass dropout probability

        }
        // Convolutional Layer
        else if (layerName == "ConvLayer") {

            // read width and activation function and then we can create the layer
            size_t depth = 0;
            TMVA::gTools().ReadAttr(layerXML, "Depth", depth);
            size_t fltHeight, fltWidth = 0;
            size_t strideRows, strideCols = 0;
            size_t padHeight, padWidth = 0;
            TMVA::gTools().ReadAttr(layerXML, "FilterHeight", fltHeight);
            TMVA::gTools().ReadAttr(layerXML, "FilterWidth", fltWidth);
            TMVA::gTools().ReadAttr(layerXML, "StrideRows", strideRows);
            TMVA::gTools().ReadAttr(layerXML, "StrideCols", strideCols);
            TMVA::gTools().ReadAttr(layerXML, "PaddingHeight", padHeight);
            TMVA::gTools().ReadAttr(layerXML, "PaddingWidth", padWidth);

            // Read activation function.
            TString funcString;
            TMVA::gTools().ReadAttr(layerXML, "ActivationFunction", funcString);
            EActivationFunction actFunction = static_cast<EActivationFunction>(funcString.Atoi());


            fNetx->AddConvLayer(depth, fltHeight, fltWidth, strideRows, strideCols,
                                padHeight, padWidth, actFunction);

        }

        // MaxPool Layer
        else if (layerName == "MaxPoolLayer") {

            // read maxpool layer info
            size_t filterHeight, filterWidth = 0;
            size_t strideRows, strideCols = 0;
            TMVA::gTools().ReadAttr(layerXML, "FilterHeight", filterHeight);
            TMVA::gTools().ReadAttr(layerXML, "FilterWidth", filterWidth);
            TMVA::gTools().ReadAttr(layerXML, "StrideRows", strideRows);
            TMVA::gTools().ReadAttr(layerXML, "StrideCols", strideCols);

            fNetx->AddMaxPoolLayer(filterHeight, filterWidth, strideRows, strideCols);
        }
        // Reshape Layer
        else if (layerName == "ReshapeLayer") {

            // read reshape layer info
            size_t depth, height, width = 0;
            TMVA::gTools().ReadAttr(layerXML, "Depth", depth);
            TMVA::gTools().ReadAttr(layerXML, "Height", height);
            TMVA::gTools().ReadAttr(layerXML, "Width", width);
            int flattening = 0;
            TMVA::gTools().ReadAttr(layerXML, "Flattening",flattening );

            fNetx->AddReshapeLayer(depth, height, width, flattening);

        }
        // RNN Layer
        else if (layerName == "RNNLayer") {

            // read RNN layer info
            size_t  stateSize,inputSize, timeSteps = 0;
            int rememberState= 0;
            int returnSequence = 0;
            TMVA::gTools().ReadAttr(layerXML, "StateSize", stateSize);
            TMVA::gTools().ReadAttr(layerXML, "InputSize", inputSize);
            TMVA::gTools().ReadAttr(layerXML, "TimeSteps", timeSteps);
            TMVA::gTools().ReadAttr(layerXML, "RememberState", rememberState );
            TMVA::gTools().ReadAttr(layerXML, "ReturnSequence", returnSequence);

            fNetx->AddBasicRNNLayer(stateSize, inputSize, timeSteps, rememberState, returnSequence);

        }
        // BatchNorm Layer
        else if (layerName == "BatchNormLayer") {
            // use some dammy value which will be overwrittem in BatchNormLayer::ReadWeightsFromXML
            fNetx->AddBatchNormLayer(0., 0.0);
        }
        // read weights and biases
        fNetx->GetLayers().back()->ReadWeightsFromXML(layerXML);
        //fNetx->GetLayers().back()->Print();
        
        // read next layer
        layerXML = TMVA::gTools().GetNextChild(layerXML);
    }
    std::cout  << "*****   Deep Learning Network  *****" << std::endl;
    fNetx->Print();

    TMVA::gTools().xmlengine().FreeDoc(model);
    return fNetx;
}