package ml.shifu.shifu.request.processor;

import com.google.common.base.Joiner;
import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.TransformationExecService;
import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.*;

import java.io.PrintWriter;
import java.util.Map;
import java.util.List;

public class ExecTransformRequestProcessor {

    private RequestObject req;
    private TransformationExecService transformationService;
    private PMML pmml;
    private String pathPMML;

    public void run(RequestObject req) throws Exception {

        this.req = req;


        String pathPMML = (String) req.getParams().get("pathPMML", "model.xml");

        pmml = PMMLUtils.loadPMML(pathPMML);

        if (req.getExecutionMode().equals(RequestObject.ExecutionMode.LOCAL_SINGLE)) {
            runLocalSingle();
        }


    }

    private void runLocalSingle() {

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<Object>> rows = loader.load(req.getParams().get("pathInputData").toString());



        Model model = PMMLUtils.getModelByName(pmml, req.getParams().get("modelName").toString());


        Map<FieldName, DerivedField> derivedFieldMap = PMMLUtils.getDerivedFieldMap(model.getLocalTransformations());


        Map<FieldName, Integer> fieldNumMap = PMMLUtils.getFieldNumMap(pmml.getDataDictionary());

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getParams().get("bindings"));
        Injector injector = Guice.createInjector(module);

        TransformationExecService service = injector.getInstance(TransformationExecService.class);

        PrintWriter writer = null;
        try {
            writer = new PrintWriter(req.getParams().get("pathOutputData").toString(), "UTF-8");

            for (List<Object> row : rows) {
                /*List<Object> transformedRow = new ArrayList<Object>();

                for (MiningField miningField : model.getMiningSchema().getMiningFields()) {

                    int fieldNum = fieldNumMap.get(miningField.getName());

                    if (miningField.getUsageType().equals(FieldUsageType.ACTIVE)) {
                        DerivedField derivedField = derivedFieldMap.get(miningField.getName());
                        transformedRow.add(service.exec(derivedField, row.get(fieldNum)));
                    } else {
                        transformedRow.add(row.get(fieldNum));
                    }
                }
                */
                writer.println(Joiner.on(",").join(service.exec(model.getMiningSchema(), derivedFieldMap, fieldNumMap, row)));
            }



        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
            }
        }


    }
}
