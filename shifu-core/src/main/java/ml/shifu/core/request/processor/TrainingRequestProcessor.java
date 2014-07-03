package ml.shifu.core.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.TrainingService;
import ml.shifu.core.request.RequestObject;
import ml.shifu.core.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.core.util.PMMLUtils;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

public class TrainingRequestProcessor implements RequestProcessor {
    private static Logger log = LoggerFactory.getLogger(TrainingRequestProcessor.class);

    public void run(RequestObject req) throws Exception {

        CSVWithHeaderLocalSingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        String pathPMML = (String) req.getGlobalParams().get("pathPMML", "model.xml");
        String pathPMMLOutput = (String) req.getGlobalParams().get("pathPMMLOutput", pathPMML);
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

        trainingService.exec(model, pmmlDataSet, req.getGlobalParams());


        log.info("Writing PMML to: " + pathPMMLOutput);
        PMMLUtils.savePMML(pmml, pathPMMLOutput);

    }
}
