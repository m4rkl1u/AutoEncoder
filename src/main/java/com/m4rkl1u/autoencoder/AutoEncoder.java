package com.m4rkl1u.autoencoder;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.ml.data.MLDataSet;

public class AutoEncoder {

    public class MLParams{
        public double[] weights;
        public ActivationFunction func;
    }
    
    private MLParams[] params;
    private MLDataSet dataset;
    
    public void setData(MLDataSet dataset) {
        this.dataset = dataset;
    }
    
    public void setFunc(ActivationFunction func){
        
    }
    
    public void train(){
        
    }
    
}
