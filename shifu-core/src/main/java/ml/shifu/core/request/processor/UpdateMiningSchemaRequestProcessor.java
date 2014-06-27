package ml.shifu.core.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UpdateMiningSchemaService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.PMMLUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.util.Map;

public class UpdateMiningSchemaRequestProcessor implements RequestProcessor {


    public void run(RequestObject req) throws Exception {


        String pathPMML = (String) req.getGlobalParams().get("pathPMML");

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        UpdateMiningSchemaService service = injector.getInstance(UpdateMiningSchemaService.class);

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        String selectedModelName = (String) req.getGlobalParams().get("modelName", null);

        for (Model model : pmml.getModels()) {
            if (selectedModelName == null || model.getModelName().equalsIgnoreCase(selectedModelName)) {
                service.updateMiningSchema(model, req.getGlobalParams());
            }
        }

        PMMLUtils.savePMML(pmml, (String) req.getGlobalParams().get("pathPMMLOutput", pathPMML));


    }
}
