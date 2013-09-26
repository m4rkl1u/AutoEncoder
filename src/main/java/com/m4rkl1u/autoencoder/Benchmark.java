package com.m4rkl1u.autoencoder;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.awt.image.DataBufferInt;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import javax.imageio.ImageIO;

import org.encog.engine.network.activation.ActivationTANH;

public class Benchmark {
    
    public static void main(String[] args) throws IOException  {
        
        //StdDraw.picture(0.5, 0.5, "src/resources/lena.jpg");
        
        AutoEncoder encoder = new AutoEncoder();
        
        File resource = new File("src/resources");
        if(resource.exists() && resource.isDirectory()) {
            for(File f : resource.listFiles()) {
                BufferedImage img = ImageIO.read(f);
                int scaleX = 50;
                int scaleY = 50;
                Image image = img.getScaledInstance(scaleX, scaleY, Image.SCALE_DEFAULT);
                BufferedImage buffered = new BufferedImage(scaleX, scaleY, BufferedImage.TYPE_INT_ARGB);
                buffered.getGraphics().drawImage(image, 0, 0 , null);
                                
                int[] pixels = ((DataBufferInt) buffered.getRaster().getDataBuffer()).getData();     
                
                double[] input = new double[pixels.length];
                for(int i = 0 ; i < pixels.length; i ++) {
                    input[i] = (double)pixels[i] / (128 * 128 * 128 * 2);
                    assert(input[i] >= -1 && input[i] <= 1);
                }
                
                System.out.println("reading file: " + f.getName() + " with size: " + input.length);
                encoder.addData(input);
            }
        }

       encoder.addLayer(new ActivationTANH(), 1600);
       encoder.addLayer(new ActivationTANH(), 900);
    } 
}
