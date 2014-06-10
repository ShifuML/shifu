package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.builtin.TransformationExecutor;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.MiningSchemaService;
import ml.shifu.shifu.di.service.TransformationExecService;
import ml.shifu.shifu.di.service.TransformationInitService;
import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.*;

import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class CreateMiningSchemaRequestProcessor {

    private RequestObject req;
    private TransformationExecService transformationService;
    private PMML pmml;
    private String pathPMML;

    public void run(RequestObject req) throws Exception {

        this.req = req;

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>)req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        MiningSchemaService service = injector.getInstance(MiningSchemaService.class);

        PMML pmml = PMMLUtils.loadPMML((String) req.getGlobalParams().get("pathPMML"));

        String selectedModelName = (String) req.getGlobalParams().get("modelName", null);

        for (Model model : pmml.getModels()) {
            if (selectedModelName == null || model.getModelName().equalsIgnoreCase(selectedModelName)) {
                model.setMiningSchema(service.createMiningSchema(model, pmml, req));
            }
        }

        PMMLUtils.savePMML(pmml, (String) req.getGlobalParams().get("pathPMMLOutput", pathPMML));






    /*
        String pathPMML = (String)req.getGlobalParams().get("pathPMML", "model.xml");

        pmml = PMMLUtils.loadPMML(pathPMML);

        DataDictionary dict = pmml.getDataDictionary();

        Model model = pmml.getModels().get(0);

        LocalTransformations localTransformations = new LocalTransformations();

        int size = dict.getNumberOfFields();
        for (int i = 0; i < size - 1; i++) {

            DataField dataField = dict.getDataFields().get(i);

            SimpleModule module = new SimpleModule();
            module.setBindings((Map<String, String>)req.getFieldParams(dataField.getName().getValue()).get("bindings"));
            Injector injector = Guice.createInjector(module);
            TransformationInitService transformationInitService = injector.getInstance(TransformationInitService.class);

            DerivedField derivedField = transformationInitService.exec(dataField, model.getModelStats());

            localTransformations.withDerivedFields(derivedField);
        }

        model.setLocalTransformations(localTransformations);

        PMMLUtils.savePMML(pmml, "model.xml");        */
        /*
        if (req.getExecutionMode().equals(RequestObject.ExecutionMode.LOCAL_SINGLE)) {
            runLocalSingle();
        }

        runLocalSingle(); */
    }

    private void runLocalSingle() {

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<String>> rows = loader.load("src/test/resources/unittest/DataSet/iris/iris.csv");

        List<List<Object>> transformed = new ArrayList<List<Object>>();

        List<DerivedField> derivedFields =  pmml.getModels().get(0).getLocalTransformations().getDerivedFields();
        int size = derivedFields.size();

        for (List<String> row : rows) {
            List<Object> transformedRow = new ArrayList<Object>();
            for (int i = 0; i < size; i++) {
                DerivedField derivedField = derivedFields.get(i);

                transformedRow.add(TransformationExecutor.transform(derivedField, row.get(i)));
            }
            transformed.add(transformedRow);
        }

        PrintWriter writer = null;
        try {
            writer = new PrintWriter("output.txt", "UTF-8");
            writer.println(transformed);
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (writer != null) {
                writer.close();
            }
        }




    }
}
