package ml.shifu.core.request.processor.deprecated;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.PMMLMiningSchemaService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.PMMLUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.util.Map;

public class CreateMiningSchemaRequestProcessor {


    public static void run(RequestObject req) throws Exception {

    /*
        String pathPMML = (String) req.getGlobalParams().get("pathPMML");

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        PMMLMiningSchemaService service = injector.getInstance(PMMLMiningSchemaService.class);

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        String selectedModelName = (String) req.getGlobalParams().get("modelName", null);

        for (Model model : pmml.getModels()) {
            if (selectedModelName == null || model.getModelName().equalsIgnoreCase(selectedModelName)) {
                model.setMiningSchema(service.createMiningSchema(model, pmml, req));
            }
        }

        PMMLUtils.savePMML(pmml, (String) req.getGlobalParams().get("pathPMMLOutput", pathPMML));
      */

    }
}
