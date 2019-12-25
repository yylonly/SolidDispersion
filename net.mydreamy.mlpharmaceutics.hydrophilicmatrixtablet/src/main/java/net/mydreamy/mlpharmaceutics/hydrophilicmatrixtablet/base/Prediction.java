package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base;


import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.util.Arrays;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Prediction for testing
 * @author yylonly
 *
 */
public class Prediction {
	
	public static Logger log = LoggerFactory.getLogger(Prediction.class);
	
	/**
	 * 
	 * @param dataset
	 * @param size
	 * @param model
	 * @param print
	 */
	public static void prediction(String problem, String dataset, int size, String model, boolean print) {
		
		  RecordReader recordReadertest = new CSVRecordReader(1,",");
	        try {
	        	recordReadertest.initialize(new FileSplit(new ClassPathResource(dataset).getFile()));
			} catch (Exception e) {
				e.printStackTrace();
			} 

	        DataSetIterator iteratortest = null;
	        switch (problem) {
	        	case "SRMT":  iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 18, 21, true);  break;
	        	case "SRMT-craft":  iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 21, 24, true);  break;
	        	case "SRMT-ORD":  iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 18, 21, true);  break;
	        	case "OFDT":  iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 26, 26, true);  break;
	        	case "OFDF":  iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 18, 18, true);  break;
	        	case "SD":    iteratortest = new RecordReaderDataSetIterator(recordReadertest, size, 18, 18+3, true); break; 
	        }

	        DataSet alldata = iteratortest.next();
	              
	        INDArray featuresTest = alldata.getFeatureMatrix();
	        INDArray lablesTest = alldata.getLabels();
	       
	        MultiLayerNetwork bestModel = null;

	        try {
	        	bestModel = ModelSerializer.restoreMultiLayerNetwork(new File(model));
	        } 
	        catch (IOException e) {
	       		e.printStackTrace();
	       	}
	          
	        INDArray PredictionTest = bestModel.output(featuresTest);
	        
	        
	        if (print == true) {
	      
		        log.info("label value: \n" + lablesTest);
		        log.info("prediction value: \n" + PredictionTest.toString());	        	
	        }

	        //compute f2 score
	        log.info("Dataset: " + dataset);
	        log.info("Dataset size:" + String.valueOf(size));
	        log.info("Model:" + dataset);
	        
	        switch (problem) {
	        	case "SRMT":  Evaluation.f2(lablesTest, PredictionTest); break;
	        	case "SRMT-craft":  Evaluation.f2(lablesTest, PredictionTest); break;
	        	case "SRMT-ORD" :  Evaluation.AverageAccuracyR(lablesTest, PredictionTest, new RegressionEvaluation(4)); break;
	        	case "OFDT":   Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.10); break;
	        	case "OFDF":  Evaluation.AccuracyMAE(lablesTest, PredictionTest, 0.10);	 break;
	        	case "SD": break;
	        }

	}

}
