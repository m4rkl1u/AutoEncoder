package com.m4rkl1u.autoencoder;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;

public class AutoEncoder {

    public class MLParams{
        public double[] weights;
        public ActivationFunction func;
        
        public MLParams(double[] weights, ActivationFunction func){
            this.weights = weights;
            this.func    = func;
        }
    }
    
    private List<MLParams> params;
    private MLDataSet dataset;
    private ActivationFunction func;
    
    private BasicNetwork network;
    private BasicNetwork hiddenNet;
    private MLDataSet intermediateDataset;
    
    public AutoEncoder(){
        params  = new ArrayList<MLParams>();
        dataset = new BasicMLDataSet();
    }
    
    public void setData(MLDataSet dataset) {
        this.dataset = dataset;
    }
    
    public void setData(double[][] p) {
        for(int i = 0 ; i < p.length; i ++){
            double[] input  = p[i];
            
            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(input));
            
            dataset.add(pair);
        }
    }
    
    public void setFunc(ActivationFunction func){
        this.func = func;
    }
    
    public void addLayer(ActivationFunction func, int nodes){
        
        buildNetwork();
        
        transformData();
        
        network = new BasicNetwork();
        
        network.addLayer(new BasicLayer(new ActivationLinear(), true, intermediateDataset.getInputSize()));
        
        network.addLayer(new BasicLayer(func, true, nodes));
        
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, intermediateDataset.getIdealSize()));
        
        network.getStructure().finalizeStructure();
        
        network.reset();
        
        train();
    }
    
    public void train() {
        Propagation propagation = new QuickPropagation(network, intermediateDataset, 0.01);
        
        propagation.setThreadCount(0);
        
        for(int i = 0 ; i < 100; i ++) {
            propagation.iteration();
            
            System.out.println( "In deep layer:" + params.size() + "Training error " + propagation.getError());
        }
        
        
        int fromNodes = network.getInputCount() + 1;
        int toNodes = network.getLayerNeuronCount(1); //the next layer
        
        int numWeight = fromNodes * toNodes;
        
        double[] weights = new double[numWeight];
        
        for(int i = 0 ; i < fromNodes; i ++ ){
            for(int j = 0 ; j < toNodes; j ++) {
                //FIXME, bug
                weights[i * fromNodes + j] = network.getWeight(0, i, j);
            }
        }
        
        ActivationFunction func = network.getActivation(1);
        
        
        MLParams param = new MLParams(weights, func);
        
        params.add(param);
        
        System.out.println("Add weight: " + Arrays.toString(weights) + "\n and the activation function: " + func.toString());
    }

    private void transformData() {
        intermediateDataset = new BasicMLDataSet();
        for(int i = 0 ; i < this.dataset.getRecordCount(); i ++) {
            MLData input  = hiddenNet.compute(dataset.get(i).getInput());
            intermediateDataset.add(input, input);
        }
    }

    private void buildNetwork() {

        hiddenNet = new BasicNetwork();
        
        hiddenNet.addLayer(new BasicLayer(new ActivationLinear(), true, dataset.getInputSize()));
        
        for(int i = 0 ; i < params.size(); i ++ ){
            hiddenNet.addLayer(new BasicLayer(params.get(i).func, true, params.get(i).weights.length));
        }
        
        hiddenNet.getStructure().finalizeStructure();
        
        for(int i = 0 ; i < params.size(); i ++) {
            double[] layer_weights = params.get(i).weights;
            int j = 0;
            int fromCount = network.getLayerTotalNeuronCount(i);
            int toCount   = network.getLayerNeuronCount(i + 1);
            
            for(int fromNeuron = 0; fromNeuron < fromCount; fromNeuron++){
                for(int toNeuron = 0; toNeuron < toCount; toNeuron++){
                    network.setWeight(i, fromNeuron, toNeuron, layer_weights[j++]);
                }
            }
        }
        
    }
    
    public void represent(){
        
    }
    
}
