package ml.shifu.core.di.builtin.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UpdateMiningSchemaService;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.core.util.RequestUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

public class UpdateMiningSchemaRequestProcessor implements RequestProcessor {

    public void exec(Request req) throws Exception {

        Params params = req.getProcessor().getParams();

        String pathPMML = (String) params.get("pathPMML");

        SimpleModule module = new SimpleModule();
        module.set(req);
        Injector injector = Guice.createInjector(module);
        UpdateMiningSchemaService service = injector.getInstance(UpdateMiningSchemaService.class);

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        String selectedModelName = (String) params.get("modelName", null);

        for (Model model : pmml.getModels()) {
            if (selectedModelName == null || model.getModelName().equalsIgnoreCase(selectedModelName)) {
                service.updateMiningSchema(model, RequestUtils.getBindingParamsBySpi(req, "PMMLMiningSchemaUpdater"));
            }
        }

        PMMLUtils.savePMML(pmml, (String) params.get("pathPMMLOutput", pathPMML));
    }
}
