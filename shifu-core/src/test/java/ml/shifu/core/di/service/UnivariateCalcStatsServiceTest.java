package ml.shifu.core.di.service;


import org.dmg.pmml.DataDictionary;
import org.testng.annotations.Test;

import java.util.List;

public class UnivariateCalcStatsServiceTest {

    private DataDictionary dict;
    private List<List<String>> rows;
    private List<List<String>> columns;

    @Test
    public void test() {
        /*
        SimpleModule module = new SimpleModule();

        //module.set("UnivariateStatsCalculator", "ml.core.core.di.builtin.stats.SimpleUnivariateStatsCalculator");
        module.set("UnivariateStatsCalculator", "ml.core.core.di.builtin.stats.BinomialUnivariateStatsCalculator");

        Injector injector = Guice.createInjector(module);

        UnivariateStatsService service = injector.getInstance(UnivariateStatsService.class);


        loadData();

        Params params = new Params();

        params.put("tags", columns.get(4));
        params.put("numBins", 10);
        params.put("posTags", Arrays.asList("Iris-setosa", "Iris-versicolor"));
        params.put("negTags", Arrays.asList("Iris-virginica"));

        PMML pmml = new PMML();
        Model model = new NeuralNetwork();
        ModelStats modelStats = new ModelStats();

        int size = dict.getNumberOfFields();
        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<String> column = columns.get(i);

            UnivariateStats univariateStats = service.getUnivariateStats(field, column, params);

            modelStats.withUnivariateStats(univariateStats);

        }


        OutputStream os = null;

        model.setModelStats(modelStats);
        pmml.withModels(model);

        try {
            os = new FileOutputStream("test3.xml");
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);

        } catch (Exception e) {
            e.printStackTrace();
        }                                   */

    }

    private void loadData() {
        /*
        TripletDataDictionaryInitializer initializer = new TripletDataDictionaryInitializer();

        Params params = new Params();


        params.put("filePath", "src/test/resources/conf/IrisFields.txt");

        dict = initializer.init(params);

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        rows = loader.load("src/test/resources/unittest/DataSet/iris/iris.csv");

        columns = LocalDataTransposer.transpose(rows);

                                                      */
    }
}
