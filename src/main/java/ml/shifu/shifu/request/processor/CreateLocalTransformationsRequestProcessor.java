package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.DerivedFieldService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

import java.util.Map;

public class CreateLocalTransformationsRequestProcessor {

    public static void run(RequestObject req) throws Exception {

        String pathPMML = (String)req.getParams().get("pathPMML", "model.xml");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        DataDictionary dataDictionary = pmml.getDataDictionary();

        Model model = PMMLUtils.getModelByName(pmml, req.getParams().get("modelName").toString());

        LocalTransformations localTransformations = new LocalTransformations();

        SimpleModule module = new SimpleModule();
        Injector injector;

        for (DataField dataField : dataDictionary.getDataFields()) {

            String fieldNameString = dataField.getName().getValue();
            Params fieldParams = req.getFieldParams(fieldNameString);
            if (fieldParams.containsKey("bindings")) {
                module.setBindings((Map<String, String>)fieldParams.get("bindings"));
                injector = Guice.createInjector(module);
                DerivedFieldService derivedFieldService = injector.getInstance(DerivedFieldService.class);
                DerivedField derivedField = derivedFieldService.exec(dataField, model.getModelStats());

                localTransformations.withDerivedFields(derivedField);
            }

        }

        model.setLocalTransformations(localTransformations);

        PMMLUtils.savePMML(pmml, (String)req.getParams().get("pathPMMLOutput", pathPMML));

    }
}
