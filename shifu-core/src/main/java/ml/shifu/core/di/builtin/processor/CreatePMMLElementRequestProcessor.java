package ml.shifu.core.di.builtin.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.builtin.ShifuPMMLCreator;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.*;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.core.util.RequestUtils;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Targets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

public class CreatePMMLElementRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory.getLogger(CreatePMMLElementRequestProcessor.class);

    public void exec(Request req) throws Exception {


        Binding processorBinding = req.getProcessor();

        Params params = processorBinding.getParams();

        String pathPMML = params.get("pathPMML", "./model.xml").toString();

        PMML pmml;

        if ((new File(pathPMML)).exists()) {
            pmml = PMMLUtils.loadPMML(pathPMML);
        } else {
            pmml = (new ShifuPMMLCreator()).create(params);
        }


        // Set bindings in module
        SimpleModule module = new SimpleModule();

        Binding dataDictionaryCreatorBinding = RequestUtils.getUniqueBinding(req, "PMMLDataDictionaryCreator");
        module.set(dataDictionaryCreatorBinding);

        Binding modelElementCreatorBinding = RequestUtils.getUniqueBinding(req, "PMMLModelElementCreator");
        module.set(modelElementCreatorBinding);

        Binding targetsCreatorBinding = RequestUtils.getUniqueBinding(req, "PMMLTargetsCreator");
        module.set(targetsCreatorBinding);

        //module.set(RequestUtils.getUniqueBinding(req, "OutputCreator", false));

        Binding miningSchemaBinding = RequestUtils.getUniqueBinding(req, "PMMLMiningSchemaCreator");
        if (miningSchemaBinding != null) {
            module.set(miningSchemaBinding);
        }


        Binding localTransformationsBinding = RequestUtils.getBindingBySpi(req, "PMMLLocalTransformationsCreator");
        if (localTransformationsBinding != null) {
            module.set(localTransformationsBinding);
        }

        // Inject
        Injector injector = Guice.createInjector(module);

        if (module.has("PMMLDataDictionaryCreator")) {
            PMMLDataDictionaryService service = injector.getInstance(PMMLDataDictionaryService.class);
            DataDictionary dataDictionary = service.getDataDictionary(dataDictionaryCreatorBinding.getParams());
            pmml.setDataDictionary(dataDictionary);
        }

        if (module.has("PMMLModelElementCreator")) {
            PMMLModelElementService service = injector.getInstance(PMMLModelElementService.class);
            Model model = service.getModelElement(modelElementCreatorBinding.getParams());
            pmml.withModels(model);
        }

        if (module.has("PMMLTargetsCreator")) {
            PMMLTargetsService service = injector.getInstance(PMMLTargetsService.class);
            Targets targets = service.createTargetsElement(pmml, targetsCreatorBinding.getParams());
        }

        if (module.has("PMMLMiningSchemaCreator")) {
            PMMLMiningSchemaService service = injector.getInstance(PMMLMiningSchemaService.class);
            service.createMiningSchema(pmml, miningSchemaBinding != null ? miningSchemaBinding.getParams() : null);
        }


        if (module.has("PMMLLocalTransformationsCreator")) {
            PMMLLocalTransformationsService service = injector.getInstance(PMMLLocalTransformationsService.class);

            service.createLocalTransformations(pmml, localTransformationsBinding != null ? localTransformationsBinding.getParams() : null);

        }

        PMMLUtils.savePMML(pmml, pathPMML);
    }

}
