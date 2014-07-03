package ml.shifu.core.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.builtin.derivedField.PassThroughDerivedFieldCreator;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.DerivedFieldService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.*;

import java.util.Map;

public class CreateLocalTransformationsRequestProcessor {

    public static void run(RequestObject req) throws Exception {

        String pathPMML = (String) req.getGlobalParams().get("pathPMML", "model.xml");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        DataDictionary dataDictionary = pmml.getDataDictionary();

        Model model = PMMLUtils.getModelByName(pmml, req.getGlobalParams().get("modelName").toString());

        LocalTransformations localTransformations = new LocalTransformations();

        SimpleModule module = new SimpleModule();
        Injector injector;

        for (DataField dataField : dataDictionary.getDataFields()) {

            String fieldNameString = dataField.getName().getValue();
            Params fieldParams = req.getFieldParams(fieldNameString);
            if (fieldParams.containsKey("bindings")) {
                module.setBindings((Map<String, String>) fieldParams.get("bindings"));
                injector = Guice.createInjector(module);
                DerivedFieldService derivedFieldService = injector.getInstance(DerivedFieldService.class);
                DerivedField derivedField = derivedFieldService.exec(dataField, model.getModelStats(), fieldParams);

                localTransformations.withDerivedFields(derivedField);
            } else {
                PassThroughDerivedFieldCreator creator = new PassThroughDerivedFieldCreator();
                DerivedField derivedField = creator.create(dataField, null, fieldParams);
                localTransformations.withDerivedFields(derivedField);
            }


        }

        model.setLocalTransformations(localTransformations);

        PMMLUtils.savePMML(pmml, (String) req.getGlobalParams().get("pathPMMLOutput", pathPMML));

    }
}
