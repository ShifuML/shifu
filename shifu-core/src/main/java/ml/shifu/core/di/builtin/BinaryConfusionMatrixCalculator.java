package ml.shifu.core.di.builtin;

import ml.shifu.core.container.BinaryConfusionMatrix;
import ml.shifu.core.container.ClassificationResult;

import java.util.ArrayList;
import java.util.List;

public class BinaryConfusionMatrixCalculator {



    public List<BinaryConfusionMatrix> calculate(List<ClassificationResult> classificationResults, List<String> posTags, List<String> negTags) {

        Double posScaleFactor = 1.0;
        Double negScaleFactor = 1.0;
        List<BinaryConfusionMatrix> confusionMatrixList = new ArrayList<BinaryConfusionMatrix>();

        Double sumPos = 0.0;
        Double sumNeg = 0.0;
        Double weightedSumPos = 0.0;
        Double weightedSumNeg = 0.0;


        // First Pass: calculate sum
        for (ClassificationResult classificationResult : classificationResults) {
            if (posTags.contains(classificationResult.getTrueClass())) {
                sumPos += posScaleFactor;
                weightedSumPos += posScaleFactor * classificationResult.getWeight();
            } else {
                // how to deal with unseen tags?
                sumNeg += negScaleFactor;
                weightedSumNeg += negScaleFactor * classificationResult.getWeight();
            }
        }


        Double tp = 0.0;
        Double fp = 0.0;
        Double weightedTp = 0.0;
        Double weightedFp = 0.0;

        BinaryConfusionMatrix confusionMatrix = new BinaryConfusionMatrix();
        confusionMatrix.setTp(0.0);
        confusionMatrix.setWeightedTp(0.0);
        confusionMatrix.setFp(0.0);
        confusionMatrix.setWeightedFp(0.0);
        confusionMatrix.setFn(sumPos);
        confusionMatrix.setWeightedFn(weightedSumPos);
        confusionMatrix.setTn(sumNeg);
        confusionMatrix.setWeightedTn(weightedSumNeg);
        confusionMatrixList.add(confusionMatrix);

        // Second Pass: calculate ConfusionMatrix
        for (ClassificationResult classificationResult : classificationResults) {
            confusionMatrix = new BinaryConfusionMatrix();

            if (posTags.contains(classificationResult.getTrueClass())) {
                tp += posScaleFactor;
                weightedTp += posScaleFactor * classificationResult.getWeight();
            } else {
                // how to deal with unseen tags?
                fp += negScaleFactor;
                weightedFp += negScaleFactor * classificationResult.getWeight();
            }

            confusionMatrix.setTp(tp);
            confusionMatrix.setWeightedTp(weightedTp);
            confusionMatrix.setFp(fp);
            confusionMatrix.setWeightedFp(weightedFp);
            confusionMatrix.setFn(sumPos - tp);
            confusionMatrix.setWeightedFn(weightedSumPos - weightedTp);
            confusionMatrix.setTn(sumNeg - fp);
            confusionMatrix.setWeightedTn(weightedSumNeg - weightedFp);
            confusionMatrixList.add(confusionMatrix);
        }

        for (BinaryConfusionMatrix matrix : confusionMatrixList) {
            matrix.calculateActionRate();
            matrix.calculateFalsePositiveRate();
            matrix.calculatePrecision();
            matrix.calculateRecall();
        }

        return confusionMatrixList;
    }


}
