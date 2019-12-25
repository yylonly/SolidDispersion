package net.mydreamy.mlpharmaceutics.hydrophilicmatrixtablet.base;

import org.nd4j.linalg.factory.Nd4j;

public class TestPrediction {

	public static void main(String[] args) {
		// TODO Auto-generated method stub

		double[] p1 = {0.2,0.33,0.42,0.52};
		double[] t1 = {0.19,0.3,0.39,0.49};

		Evaluation.f2(Nd4j.create(p1), Nd4j.create(t1));
		

		double[] p2 = {0.35, 0.53, 0.70, 0.84};
		double[] t2 = {0.41, 0.59, 0.74, 0.79};

		Evaluation.f2(Nd4j.create(p2), Nd4j.create(t2));
				
		
	}

}
