package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.MiningSchemaService;
import ml.shifu.shifu.di.service.UpdateMiningSchemaService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.util.Map;

public class UpdateMiningSchemaRequestProcessor implements RequestProcessor{



    public void run(RequestObject req) throws Exception {


        String pathPMML = (String) req.getParams().get("pathPMML");

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        UpdateMiningSchemaService service = injector.getInstance(UpdateMiningSchemaService.class);

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        String selectedModelName = (String) req.getParams().get("modelName", null);

        for (Model model : pmml.getModels()) {
            if (selectedModelName == null || model.getModelName().equalsIgnoreCase(selectedModelName)) {
                service.updateMiningSchema(model, req.getParams());
            }
        }

        PMMLUtils.savePMML(pmml, (String) req.getParams().get("pathPMMLOutput", pathPMML));


    }
}
