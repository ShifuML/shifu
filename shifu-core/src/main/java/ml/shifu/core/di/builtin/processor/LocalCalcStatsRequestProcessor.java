package ml.shifu.core.di.builtin.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.core.di.module.SimpleModule;
import ml.shifu.core.di.service.UnivariateStatsService;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.di.spi.SingleThreadFileLoader;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.*;
import org.dmg.pmml.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class LocalCalcStatsRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory.getLogger(LocalCalcStatsRequestProcessor.class);

    public void exec(Request req) throws Exception {
        SimpleModule module = new SimpleModule();

        Binding dataDictionaryCreatorBinding = RequestUtils.getUniqueBinding(req, "UnivariateStatsCalculator");
        module.set(dataDictionaryCreatorBinding);


        Injector injector = Guice.createInjector(module);
        UnivariateStatsService univariateStatsService = injector.getInstance(UnivariateStatsService.class);

        String pathPMML = req.getProcessor().getParams().get("pathPMML", "model.xml").toString();
        log.info("PMML Path: " + pathPMML);


        PMML pmml = PMMLUtils.loadPMML(pathPMML);
        Params bindingParams = dataDictionaryCreatorBinding.getParams();

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<Object>> rows = loader.load((String) bindingParams.get("pathInputData"));

        List<List<Object>> columns = LocalDataTransposer.transpose(rows);

        DataDictionary dict = pmml.getDataDictionary();

        Model model = PMMLUtils.getModelByName(pmml, (String) bindingParams.get("modelName"));

        ModelStats modelStats = new ModelStats();
        int size = dict.getNumberOfFields();

        // TODO: this is wrong, the requestprocessor should not know anything about binding params
        int targetFieldNum = PMMLUtils.getTargetFieldNumByName(pmml.getDataDictionary(), (String) bindingParams.get("targetFieldName"));


        bindingParams.put("tags", columns.get(targetFieldNum));
        
        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<Object> values = columns.get(i);

            UnivariateStats univariateStats = univariateStatsService.getUnivariateStats(field, values, bindingParams);
            modelStats.withUnivariateStats(univariateStats);
        }
        model.setModelStats(modelStats);


        PMMLUtils.savePMML(pmml, pathPMML);

    }

}
