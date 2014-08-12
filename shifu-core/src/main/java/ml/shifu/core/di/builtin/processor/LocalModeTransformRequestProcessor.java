package ml.shifu.core.di.builtin.processor;


import com.google.common.base.Joiner;
import ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.di.spi.SingleThreadFileLoader;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class LocalModeTransformRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory.getLogger(LocalModeTransformRequestProcessor.class);

    public void exec(Request req) throws Exception {
        Params params = req.getProcessor().getParams();
        String pathPMML = params.get("pathPMML", "model.xml").toString();
        String pathOutputActiveHeader = params.get("pathOutputActiveHeader").toString();


        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<Object>> rows = loader.load(params.get("pathInputData").toString());


        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName").toString());

        Map<FieldUsageType, List<DerivedField>> fieldsMap = PMMLUtils.getDerivedFieldsByUsageType(pmml, model);

        List<DerivedField> activeFields = fieldsMap.get(FieldUsageType.ACTIVE);
        List<DerivedField> targetFields = fieldsMap.get(FieldUsageType.TARGET);
        //Map<FieldName, DerivedField> derivedFieldMap = PMMLUtils.getDerivedFieldMap(model.getLocalTransformations());


        //Map<FieldName, Integer> fieldNumMap = PMMLUtils.getFieldNumMap(pmml.getDataDictionary());

        //SimpleModule module = new SimpleModule();
        //module.set(req);
        //Injector injector = Guice.createInjector(module);

        //TransformationExecService service = injector.getInstance(TransformationExecService.class);

        PrintWriter writer = null;
        PrintWriter headerWriter = null;

        DefaultTransformationExecutor executor = new DefaultTransformationExecutor();
        //AllInclusiveTransformationExecutor executor = new AllInclusiveTransformationExecutor();

        try {
            writer = new PrintWriter(params.get("pathOutputData").toString(), "UTF-8");
            headerWriter = new PrintWriter(pathOutputActiveHeader);

            List<String> header = new ArrayList<String>();
            for (DerivedField derivedField : targetFields) {
                header.add("TARGET::" + derivedField.getName().getValue());
            }
            for (DerivedField derivedField : activeFields) {
                header.add("ACTIVE::" + derivedField.getName().getValue());
            }
            headerWriter.print(Joiner.on(",").join(header));

            List<DataField> dataFields = pmml.getDataDictionary().getDataFields();
            int size = dataFields.size();
            Map<String, Object> rawDataMap = new HashMap<String, Object>();

            for (List<Object> row : rows) {

                for (int i = 0; i < size; i++) {
                    rawDataMap.put(dataFields.get(i).getName().getValue(), row.get(i));
                }
                List<Object> result = executor.transform(targetFields, rawDataMap);
                result.addAll(executor.transform(activeFields, rawDataMap));
                writer.println(Joiner.on(",").join(result));
            }


        } catch (Exception e) {
            log.error(e.toString());
        } finally {
            if (writer != null) {
                writer.close();
            }
            if (headerWriter != null) {
                headerWriter.close();
            }
        }
    }
}
