package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.manual;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base.Prediction;

public class FinalTestResult {

	public static void main(String[] args) {
		
		
		//best Model testing
		Prediction.prediction("SRMT-ORD", "manual/trainingset.csv", 200, "src/main/resources/manual/bestModel.bin", false); 
		Prediction.prediction("SRMT-ORD", "manual/testingset.csv", 20, "src/main/resources/manual/bestModel.bin", false); 
		
	}
}
