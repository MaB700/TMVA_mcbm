#include <string>
#include <iostream>
#include <vector>
#include "TMVA/Reader.h"
#include "TFile.h"
#include "TTree.h"
#include "TMVA/MethodDL.h"
#include "TStopwatch.h"


using namespace TMVA::DNN::CNN;
using namespace TMVA::DNN;

using TMVA::DNN::EActivationFunction;
using TMVA::DNN::ELossFunction;
using TMVA::DNN::EInitialization;
using TMVA::DNN::EOutputFunction;
using TMVA::DNN::EOptimizer;

using Architecture_t = TMVA::DNN::TCpu<Float_t>;
using Scalar_t = typename Architecture_t::Scalar_t;
using Matrix_t = typename Architecture_t::Matrix_t;
using Tensor_t = typename Architecture_t::Tensor_t;
using Layer_t = TMVA::DNN::VGeneralLayer<Architecture_t>;
using DeepNet_t = TMVA::DNN::TDeepNet<Architecture_t, Layer_t>;
using TMVAInput_t =  std::tuple<const std::vector<TMVA::Event *> &, const TMVA::DataSetInfo &>;
using TensorDataLoader_t = TTensorDataLoader<TMVAInput_t, Architecture_t>;

DeepNet_t* ReadModelFromXML(TString);

int predict_comp(){
    
    // predict with reader class ---------------------------
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
    std::vector<Float_t> out;
    out = reader->EvaluateRegression("DL");

    // predict with lowlevel functions ---------------------
    TString fileXML = "dataset/weights/mCBM_tmvaDL.weights.xml";
    DeepNet_t* fNet = ReadModelFromXML(fileXML);
    
    /* for(int i{}; i < matrix.GetNrows(); i++){
        for(int j{}; j < matrix.GetNcols(); j++){
            std::cout << matrix(i,j) << " " ;
        }
        std::cout << std::endl;
    } */
    
    
    Matrix_t* pred = new TCpuMatrix<Float_t>(1, 72*32);
    //Tensor_t* input = new TCpuTensor<Float_t>(1, 1, 72, 32);
    Tensor_t input = Architecture_t::CreateTensor(fNet->GetBatchSize(), fNet->GetInputDepth(), fNet->GetInputHeight(), fNet->GetInputWidth() ); //TODO: store on heap
    TCpuBuffer<Float_t>* inputBuffer = 
                    new TCpuBuffer<Float_t>( input.GetSize());
    input.Zero();
    pred->Zero();
    //Float_t* data = *(input->GetContainer());
    for(int i{}; i < 2304; i++){
        (*inputBuffer)[i] = in[i];
    }
    input.GetDeviceBuffer().CopyFrom(*inputBuffer);
    fNet->Prediction(*pred, input, TMVA::DNN::EOutputFunction::kIdentity);
    std::cout << "index " << "reader " << "lowlevel " << std::endl;
    for(int i{}; i < 2304; i++){
        if(out[i] > 0.1 || (*pred)(0,i) > 0.1 ){
        std::cout << i << " " << out[i] << " " << (*pred)(0,i) << std::endl;
        }
    }



    return 1;

}


DeepNet_t* ReadModelFromXML(TString xmlFile){

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
    DeepNet_t* fNetx = new DeepNet_t(1, inputDepth, inputHeight, inputWidth, batchDepth, batchHeight, batchWidth, 
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