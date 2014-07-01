package ml.shifu.core.request.processor;

import ml.shifu.core.di.builtin.executor.PMMLExecutor;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.LocalDataUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.PMML;

import java.util.HashMap;
import java.util.Map;
import java.util.List;

public class ModelExecutionProcessor {

    public void run(RequestObject req) throws Exception {
        Params globalParams = req.getGlobalParams();
        String pathHeader = (String) globalParams.get("pathHeader");
        String pathData = (String) globalParams.get("pathData");

        String pathPMML = (String) globalParams.get("pathPMML");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        List<String> header = LocalDataUtils.loadHeader(pathHeader, "|");
        List<List<Object>> data = LocalDataUtils.loadData(pathData, ",");

        PMMLExecutor executor = new PMMLExecutor(pmml);


        String targetFieldName = pmml.getModels().get(0).getTargets().getTargets().get(0).getField().getValue();

        for (List<Object> row : data) {
            Map<String, Object> rawDataMap = createRawDataMap(header, row);
            Object result = executor.exec(rawDataMap);

            System.out.println(rawDataMap.get(targetFieldName) + ", " + result);
        }
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
