package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.container.PMMLDataSet;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.TrainingService;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;

import java.util.List;
import java.util.Map;

public class TrainingRequestProcessor implements RequestProcessor {

    public void run(RequestObject req) throws Exception {
        CSVWithHeaderLocalSingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        String pathPMML = (String) req.getGlobalParams().get("pathPMML", "model.xml");
        PMML pmml = PMMLUtils.loadPMML(pathPMML);
        Model model = PMMLUtils.getModelByName(pmml, req.getGlobalParams().get("modelName").toString());

        String pathNormalizedData = (String) req.getGlobalParams().get("pathNormalizedData");

        List<List<Object>> data = loader.load(pathNormalizedData);

        PMMLDataSet pmmlDataSet = new PMMLDataSet();
        pmmlDataSet.setRows(data);
        pmmlDataSet.setMiningSchema(model.getMiningSchema());

        SimpleModule module = new SimpleModule();
        module.setBindings((Map<String, String>) req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);

        TrainingService trainingService = injector.getInstance(TrainingService.class);

        trainingService.exec(pmmlDataSet, req.getGlobalParams());

    }
}
