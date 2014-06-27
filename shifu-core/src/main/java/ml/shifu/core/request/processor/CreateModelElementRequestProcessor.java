package ml.shifu.core.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.ModelElementService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.dmg.pmml.Targets;

import java.util.Map;

public class CreateModelElementRequestProcessor {


    public static void run(RequestObject req) throws Exception {

        String pathPMML = (String) req.getGlobalParams().get("pathPMML", "model.xml");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        Params globalParams = req.getGlobalParams();


        Model model = PMMLUtils.createModelByType(globalParams.get("modelType").toString());

        model.setModelName(globalParams.get("modelName").toString());

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) globalParams.get("bindings"));

        Injector injector = Guice.createInjector(module);

        ModelElementService service = injector.getInstance(ModelElementService.class);


        Targets targets = service.getTargetsElement(globalParams);

        model.setTargets(targets);
        pmml.withModels(model);

        PMMLUtils.savePMML(pmml, (String) globalParams.get("pathPMMLOutput", pathPMML));
    }

}
