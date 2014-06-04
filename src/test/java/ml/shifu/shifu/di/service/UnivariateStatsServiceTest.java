package ml.shifu.shifu.di.service;


import com.google.inject.Guice;
import com.google.inject.Injector;
import ml.shifu.shifu.di.builtin.SimpleUnivariateStatsCalculator;
import ml.shifu.shifu.di.builtin.TripletDataDictionaryInitializer;
import ml.shifu.shifu.di.module.SimpleModule;
import ml.shifu.shifu.di.spi.SingleThreadFileLoader;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.util.CSVWithHeaderLocalSingleThreadFileLoader;
import ml.shifu.shifu.util.LocalDataTransposer;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;
import org.jpmml.model.JAXBUtil;
import org.testng.annotations.Test;

import javax.xml.transform.stream.StreamResult;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class UnivariateStatsServiceTest {

    @Test
    public void test() {
        SimpleModule module = new SimpleModule();

        //module.set("UnivariateStatsCalculator", "ml.shifu.shifu.di.builtin.SimpleUnivariateStatsCalculator");
        module.set("UnivariateStatsCalculator", "ml.shifu.shifu.di.builtin.BinomialUnivariateStatsCalculator");

        Injector injector = Guice.createInjector(module);

        UnivariateStatsService service = injector.getInstance(UnivariateStatsService.class);


        loadData();

        Params params = new Params();

        params.set("tags", columns.get(4));
        params.set("numBins", 10);
        params.set("posTags", Arrays.asList("Iris-setosa", "Iris-versicolor"));
        params.set("negTags", Arrays.asList("Iris-virginica"));

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
        }

    }

    private DataDictionary dict;
    private List<List<String>> rows;
    private List<List<String>> columns;

    private void loadData() {
        TripletDataDictionaryInitializer initializer = new TripletDataDictionaryInitializer();

        Params params = new Params();


        params.set("filePath", "src/test/resources/conf/IrisFields.txt");

        dict = initializer.init(params);

        SingleThreadFileLoader loader = new CSVWithHeaderLocalSingleThreadFileLoader();

        rows = loader.load("src/test/resources/unittest/DataSet/iris/iris.csv");

        columns = LocalDataTransposer.transpose(rows);


    }
}
