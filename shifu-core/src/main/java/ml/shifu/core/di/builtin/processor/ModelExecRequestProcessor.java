package ml.shifu.core.di.builtin.processor;

import ml.shifu.core.container.ClassificationResult;
import ml.shifu.core.di.builtin.executor.PMMLModelExecutor;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.JSONUtils;
import ml.shifu.core.util.LocalDataUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.io.File;
import java.util.*;

public class ModelExecRequestProcessor implements RequestProcessor {

    public void exec(Request req) throws Exception {
        Params params = req.getProcessor().getParams();
        String pathHeader = (String) params.get("pathHeader");
        String pathData = (String) params.get("pathData");

        String pathPMML = (String) params.get("pathPMML");

        String modelName = (String) params.get("modelName");
        String pathResult = (String) params.get("pathResult");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        List<String> header = LocalDataUtils.loadHeader(pathHeader, "|");
        List<List<Object>> data = LocalDataUtils.loadData(pathData, ",");

        PMMLModelExecutor executor = new PMMLModelExecutor(pmml);

        Model model = PMMLUtils.getModelByName(pmml, modelName);

        String targetFieldName = model.getTargets().getTargets().get(0).getField().getValue();

        List<ClassificationResult> classificationResultList = new ArrayList<ClassificationResult>();

        for (List<Object> row : data) {
            Map<String, Object> rawDataMap = createRawDataMap(header, row);
            Object result = executor.exec(rawDataMap);

            ClassificationResult classificationResult = new ClassificationResult();
            classificationResult.setTrueClass(rawDataMap.get(targetFieldName).toString());
            classificationResult.putScore(modelName, Double.valueOf(result.toString()));
            for (MiningField miningField : model.getMiningSchema().getMiningFields()) {
                if (miningField.getUsageType().equals(FieldUsageType.SUPPLEMENTARY)) {
                    String fieldName = miningField.getName().getValue();
                    classificationResult.putSupplementary(fieldName, rawDataMap.get(fieldName));
                } else if (miningField.getUsageType().equals(FieldUsageType.ANALYSIS_WEIGHT)) {
                    classificationResult.setWeight(Double.valueOf(rawDataMap.get(miningField.getName().getValue()).toString()));
                }
            }


            classificationResultList.add(classificationResult);
        }

        Collections.sort(classificationResultList, new Comparator<ClassificationResult>() {
            @Override
            public int compare(ClassificationResult a, ClassificationResult b) {
                return b.getMeanScore().compareTo(a.getMeanScore());
            }
        });

        JSONUtils.writeValue(new File(pathResult), classificationResultList);

    }

    private Map<String, Object> createRawDataMap(List<String> header, List<Object> row) {
        Map<String, Object> rawDataMap = new HashMap<String, Object>();

        int size = header.size();

        for (int i = 0; i < size; i++) {
            rawDataMap.put(header.get(i), row.get(i));
        }

        return rawDataMap;
    }

}
