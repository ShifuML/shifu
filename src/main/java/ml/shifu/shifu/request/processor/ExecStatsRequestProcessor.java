package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.UnivariateStatsService;
import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.LocalDataTransposer;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;

import java.util.Map;
import java.util.List;

public class ExecStatsRequestProcessor {

    private RequestObject req;
    private UnivariateStatsService univariateStatsService;
    private PMML pmml;
    private String pathPMML;

    public void run(RequestObject req) throws Exception {

        this.req = req;
        SimpleModule module = new SimpleModule();

        module.setBindings((Map<String, String>)req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        univariateStatsService = injector.getInstance(UnivariateStatsService.class);

        pathPMML = (String)req.getGlobalParams().get("pathPMML", "model.xml");

        pmml = PMMLUtils.loadPMML(pathPMML);

        if (req.getExecutionMode().equals(RequestObject.ExecutionMode.LOCAL_SINGLE)) {
            runLocalSingle();
        }


    }

    private void runLocalSingle() {

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<Object>> rows = loader.load((String) req.getGlobalParams().get("pathInputData"));

        List<List<Object>> columns = LocalDataTransposer.transpose(rows);

        DataDictionary dict = pmml.getDataDictionary();

        Model model = PMMLUtils.getModelByName(pmml, (String) req.getGlobalParams().get("modelName"));

        ModelStats modelStats = new ModelStats();
        int size = dict.getNumberOfFields();

        int targetFieldNum = PMMLUtils.getTargetFieldNumByName(pmml.getDataDictionary(), (String) req.getGlobalParams().get("targetFieldName"));

        Params params = new Params();
        params.put("globalParams", req.getGlobalParams());
        params.put("fieldParams", null);
        params.put("tags", columns.get(targetFieldNum));






        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<Object> column = columns.get(i);

            UnivariateStats univariateStats = univariateStatsService.getUnivariateStats(field, column, params);
            modelStats.withUnivariateStats(univariateStats);
        }
        model.setModelStats(modelStats);


        PMMLUtils.savePMML(pmml, (String)req.getGlobalParams().get("pathPMMLOutput", pathPMML));

    }

}
