package ml.shifu.core.di.builtin.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.TrainingService;
import ml.shifu.core.di.service.UpdateMiningSchemaService;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.*;
import org.dmg.pmml.Model;
import org.dmg.pmml.PMML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

public class TrainRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory.getLogger(TrainRequestProcessor.class);

    public void exec(Request req) throws Exception {

        Params processorParams = req.getProcessor().getParams();

        String pathPMML = processorParams.get("pathPMML").toString();

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        Model model = PMMLUtils.getModelByName(pmml, processorParams.get("modelName").toString());

        String pathNormalizedData = processorParams.get("pathNormalizedData").toString();

        // TODO: make this injectable
        CSVWithoutHeaderLocalSingleThreadFileLoader loader = new CSVWithoutHeaderLocalSingleThreadFileLoader();
        HeaderFileLoader headerLoader = new HeaderFileLoader();

        List<String> header = headerLoader.load(processorParams.get("pathNormalizedHeader").toString());
        List<List<Object>> data = loader.load(pathNormalizedData);

        PMMLDataSet pmmlDataSet = new PMMLDataSet();
        pmmlDataSet.setRows(data);
        pmmlDataSet.setMiningSchema(model.getMiningSchema());
        pmmlDataSet.setHeader(header);

        SimpleModule module = new SimpleModule();
        module.set(req);
        Injector injector = Guice.createInjector(module);
        TrainingService service = injector.getInstance(TrainingService.class);

        Params trainParams = RequestUtils.getBindingParamsBySpi(req, "Trainer");

        service.exec(model, pmmlDataSet, trainParams);

        String pathPMMLOutput = processorParams.get("pathPMMLOutput", pathPMML).toString();

        log.info("Writing PMML to: " + pathPMMLOutput);
        PMMLUtils.savePMML(pmml, pathPMMLOutput);

    }
}
