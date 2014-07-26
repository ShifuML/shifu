package ml.shifu.core.di.builtin.transform;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.builtin.derivedfield.PassThroughPMMLDerivedFieldCreator;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.PMMLDerivedFieldService;
import ml.shifu.core.di.spi.PMMLLocalTransformationsCreator;
import ml.shifu.core.request.FieldConf;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.core.util.RequestUtils;
import org.dmg.pmml.*;

import java.util.List;


public class DefaultLocalTransformationsCreator implements PMMLLocalTransformationsCreator {

    public LocalTransformations create(PMML pmml, Params params) throws Exception {


        DataDictionary dataDictionary = pmml.getDataDictionary();

        Model model = PMMLUtils.getModelByName(pmml, params.get("modelName").toString());

        LocalTransformations localTransformations = new LocalTransformations();

        SimpleModule module = new SimpleModule();
        Injector injector;

        List<FieldConf> fieldConfs = RequestUtils.getFieldConfs(params);

        for (DataField dataField : dataDictionary.getDataFields()) {
            String fieldNameString = dataField.getName().getValue();
            FieldConf fieldConf = RequestUtils.getFieldConfByName(fieldConfs, fieldNameString);

            // Params fieldParams = fieldConf.getParams();

            if (fieldConf.getBinding() != null) {
                module.set(fieldConf.getBinding());
                injector = Guice.createInjector(module);
                PMMLDerivedFieldService derivedFieldService = injector.getInstance(PMMLDerivedFieldService.class);
                DerivedField derivedField = derivedFieldService.exec(dataField, model.getModelStats(), fieldConf.getBinding().getParams());
                localTransformations.withDerivedFields(derivedField);
            } else {
                PassThroughPMMLDerivedFieldCreator creator = new PassThroughPMMLDerivedFieldCreator();
                DerivedField derivedField = creator.create(dataField, null, null);
                localTransformations.withDerivedFields(derivedField);
            }


        }

        model.setLocalTransformations(localTransformations);

        return localTransformations;

    }

}
