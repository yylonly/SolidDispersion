package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base;

import org.deeplearning4j.eval.RegressionEvaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Evaluation {
	
	
	public static Logger log = LoggerFactory.getLogger(Evaluation.class);
	
	public static void f2(INDArray lablesTest, INDArray PredictionTest)
	{
		
		
		
        int size = lablesTest.size(0);
              
        INDArray allf2 = Nd4j.zeros(size);
        double correctnum = 0;
        
        for (int i = 0; i < size; i++)
        {
        	INDArray labelrow = lablesTest.getRow(i);
        	INDArray perdictrow = PredictionTest.getRow(i);
        	INDArray subrow = labelrow.sub(perdictrow);

        	INDArray subrow100 = subrow.mul(100);
        	double meanerror = Transforms.pow(subrow100, 2).mean(1).getDouble(0);
        	
        	//log.info("mean error: " + String.valueOf(meanerror));
        	
        	double f2 = Math.log10(Math.pow((meanerror + 1), -0.5)*100)*50;
        	
        	//log.info("f2: "+ String.valueOf(f2) + "\n");
        	
        	allf2.putScalar(i, f2);
        	
        	if (f2 >= 50)
        	{
        		correctnum++;
        	}
	        }
        
        log.info(allf2.toString());
        log.info("F2 accurecy: " + (correctnum / size));
	}
	
	public static void AccuracyMAE(INDArray lablesTest, INDArray PredictionTest, double therdsold) {
		
		
		
        INDArray absErrorMatrix = Transforms.abs(lablesTest.sub(PredictionTest)).sum(1).div(PredictionTest.size(1));
        int size = absErrorMatrix.size(0);
		INDArray allAE = Nd4j.zeros(size);

        double correct = 0;
        for (int i = 0; i < size; i++)
        {
        	if (absErrorMatrix.getDouble(i) <= therdsold)
        	{
        		correct++;
        	}
        	allAE.putScalar(i, absErrorMatrix.getDouble(i));
        }
        log.info(allAE.toString());
        log.info("AccuracyMAE  <= " + therdsold*100 + "%: " + String.format("%.4f", correct/size));
	}
	
	public static void AverageAccuracyR(INDArray lablesTest, INDArray PredictionTest,  RegressionEvaluation evalTest) {

        
   

        evalTest.eval(lablesTest, PredictionTest);	  
        
        double AverTestR = 0;
        double AverMAE = 0;
        for (int i = 0; i < 4; i++) {
        	AverTestR += evalTest.correlationR2(i);
        	AverMAE += evalTest.meanAbsoluteError(i);
        }
        
    //    log.info("testing set MSE is: " + String.format("%.10f", (evalTest.meanSquaredError(0)+evalTest.meanSquaredError(1)+evalTest.meanSquaredError(2)+evalTest.meanSquaredError(3))/4)); 
        log.info("R Score is:  " + String.format("%.4f",  AverTestR/4));
        log.info("MAE is: " + String.format("%.4f",  AverMAE/4));
	    
	}
}
