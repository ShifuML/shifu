package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.ModelElementService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
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
