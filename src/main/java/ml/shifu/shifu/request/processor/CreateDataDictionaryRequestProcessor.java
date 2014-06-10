package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.DataDictionaryService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.DataDictionary;
import org.dmg.pmml.PMML;

import java.util.Map;

public class CreateDataDictionaryRequestProcessor {

    public void run(RequestObject req) {

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>)req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);

        Params params = req.getGlobalParams();

        DataDictionaryService dataDictionaryService= injector.getInstance(DataDictionaryService.class);
        DataDictionary dataDictionary = dataDictionaryService.getDataDictionary(req);



        //MiningSchemaInitializer miningSchemaInitializer = new UseAllMiningSchemaInitializer();
        //MiningSchema miningSchema = miningSchemaInitializer.init(dataDictionary, targets, params);

        PMML pmml = new PMML();
        pmml.setDataDictionary(dataDictionary);
        //Model model = new NeuralNetwork();
        //model.setTargets(targets);
        //model.setMiningSchema(miningSchema);
        //pmml.withModels(model);

        PMMLUtils.savePMML(pmml, (String) params.get("pathPMML"));

    }
}
