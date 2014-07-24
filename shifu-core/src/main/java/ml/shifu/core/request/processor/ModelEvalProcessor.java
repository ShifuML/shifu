package ml.shifu.core.request.processor;


import ml.shifu.core.container.BinaryConfusionMatrix;
import ml.shifu.core.container.ClassificationResult;
import ml.shifu.core.di.builtin.BinaryConfusionMatrixCalculator;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.Params;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class ModelEvalProcessor {

    public void run(RequestObject req) throws Exception {

        Params params = req.getGlobalParams();

        String pathScoreResult = (String) params.get("pathScoreResult");

        List<String> posTags = (List<String>) params.get("posTags");
        List<String> negTags = (List<String>) params.get("negTags");

        List<ClassificationResult> classificationResultList = Arrays.asList(JSONUtils.readValue(new File(pathScoreResult), ClassificationResult[].class));

        BinaryConfusionMatrixCalculator calculator = new BinaryConfusionMatrixCalculator();
        List<BinaryConfusionMatrix> confusionMatrixList = calculator.calculate(classificationResultList, posTags, negTags);


        int size = confusionMatrixList.size();
        List<Double> samplePoints = (List<Double>) params.get("samplePoints");

        //JSONUtils.writeValue(new File("tmp/confusionmatrix.json"), confusionMatrixList);

        List<BinaryConfusionMatrix> sampledList = new ArrayList<BinaryConfusionMatrix>();
        for (Double samplePoint : samplePoints) {
            int index =   (int)Math.round(samplePoint * (size - 1));
            sampledList.add(confusionMatrixList.get(index));
        }

        JSONUtils.writeValue(new File("tmp/sampled.json"), sampledList);

    }
}
