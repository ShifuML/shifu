package ml.shifu.shifu.request.processor;

import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.builtin.SimpleUnivariateStatsCalculator;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.service.UnivariateStatsService;
import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.request.RequestObject;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.LocalDataTransposer;
import ml.shifu.shifu.util.PMMLUtils;
import org.dmg.pmml.*;

import java.io.IOException;
import java.util.Map;
import java.util.List;

public class StatsRequestProcessor {

    private RequestObject req;
    private UnivariateStatsService univariateStatsService;
    private PMML pmml;
    private String pathPMML;

    public void run(RequestObject req) throws Exception {
        /*
        this.req = req;
        SimpleModule module = new SimpleModule();

        module.setBindings((Map<String, String>)req.getGlobalParams().get("bindings"));
        Injector injector = Guice.createInjector(module);
        univariateStatsService = injector.getInstance(UnivariateStatsService.class);

        pathPMML = (String)req.getParams().get("pathPMML", "model.xml");

        pmml = PMMLUtils.loadPMML(pathPMML);

        if (req.getExecutionMode().equals(RequestObject.ExecutionMode.LOCAL_SINGLE)) {
            runLocalSingle();
        }
                                   */

    }

    private void runLocalSingle() {
        /*
        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        List<List<String>> rows = loader.load("src/test/resources/unittest/DataSet/iris/iris.csv");

        List<List<String>> columns = LocalDataTransposer.transpose(rows);

        DataDictionary dict = pmml.getDataDictionary();
        Model model = new NeuralNetwork();
        ModelStats modelStats = new ModelStats();
        int size = dict.getNumberOfFields();
        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<String> column = columns.get(i);

            UnivariateStats univariateStats = univariateStatsService.getUnivariateStats(field, column, req.getParams());
            modelStats.withUnivariateStats(univariateStats);
        }
        model.setModelStats(modelStats);
        pmml.withModels(model);

        PMMLUtils.savePMML(pmml, pathPMML);
                    */
    }

}
