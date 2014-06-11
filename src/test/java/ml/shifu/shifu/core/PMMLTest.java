package ml.shifu.shifu.core;

import ml.shifu.shifu.container.CategoricalValueObject;
import ml.shifu.shifu.container.NumericalValueObject;
import ml.shifu.shifu.di.builtin.stats.BinomialUnivariateStatsCalculator;
import ml.shifu.shifu.di.builtin.stats.BinomialUnivariateStatsDiscrCalculator;
import ml.shifu.shifu.di.builtin.stats.SimpleUnivariateStatsCalculator;
import ml.shifu.shifu.di.spi.UnivariateStatsCalculator;
import ml.shifu.shifu.util.JSONUtils;
import ml.shifu.shifu.util.PMMLUtils;
import ml.shifu.shifu.util.Params;
import org.dmg.pmml.*;
import org.jpmml.model.JAXBUtil;
import org.testng.Assert;
import org.testng.annotations.Test;

import javax.xml.transform.stream.StreamResult;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.util.*;

public class PMMLTest {


    @Test
    public void testOutputXML() {
        List<NumericalValueObject> nvoList = new ArrayList<NumericalValueObject>();
        for (int i = 0; i < 100; i++) {
            NumericalValueObject nvo = new NumericalValueObject();
            nvo.setValue(i * 1.0 % 39);
            nvo.setIsPositive(i % 2 == 0 ? true : false);
            nvoList.add(nvo);
        }


        List<CategoricalValueObject> cvoList = new ArrayList<CategoricalValueObject>();

        for (int i = 0; i < 100; i++) {
            CategoricalValueObject vo = new CategoricalValueObject();
            vo.setValue(i % 3 == 1 ? "Cat" : "Dog");
            vo.setIsPositive(i % 2 == 1 ? true : false);
            cvoList.add(vo);
        }




        OutputStream os = null;


        PMML pmml = new PMML();
        Model model = new NeuralNetwork();
        ModelStats modelStats = new ModelStats();

        UnivariateStats univariateStats = new UnivariateStats();
        //UnivariateStatsContCalculator.calculate(univariateStats, nvoList, 10);
        BinomialUnivariateStatsDiscrCalculator.calculate(univariateStats, cvoList, null);

        modelStats.withUnivariateStats(univariateStats);
        model.setModelStats(modelStats);
        pmml.withModels(model);

        try {
            os = new FileOutputStream("test.xml");
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testLoadData() {

        //loadData();


        Assert.assertEquals((int)dict.getNumberOfFields(), columns.size());


        Map<String, Object> statsParams = new HashMap<String, Object>();

        statsParams.put("tags", columns.get(4));
        statsParams.put("numBins", 10);
        statsParams.put("posTags", Arrays.asList("Iris-setosa", "Iris-versicolor"));
        statsParams.put("negTags", Arrays.asList("Iris-virginica"));

        PMML pmml = new PMML();
        Model model = new NeuralNetwork();
        ModelStats modelStats = new ModelStats();

        int size = dict.getNumberOfFields();
        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<String> column = columns.get(i);


            UnivariateStatsCalculator univariateStatsCalculator = new BinomialUnivariateStatsCalculator();

            //UnivariateStats univariateStats = univariateStatsCalculator.calculate(field, column, statsParams);
            //modelStats.withUnivariateStats(univariateStats);

        }


        OutputStream os = null;

        model.setModelStats(modelStats);
        pmml.withModels(model);

        try {
            os = new FileOutputStream("test.xml");
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);

        } catch (Exception e) {
            e.printStackTrace();
        }




    }

    @Test
    public void testSimpleUnivariateStats() {
        /*loadData();

        Params params = new Params();

        PMML pmml = new PMML();
        Model model = new NeuralNetwork();
        ModelStats modelStats = new ModelStats();

        int size = dict.getNumberOfFields();
        for (int i = 0; i < size; i++) {

            DataField field = dict.getDataFields().get(i);
            List<String> column = columns.get(i);


            UnivariateStatsCalculator univariateStatsCalculator = new SimpleUnivariateStatsCalculator();

            UnivariateStats univariateStats = univariateStatsCalculator.calculate(field, column, params);
            modelStats.withUnivariateStats(univariateStats);

        }


        OutputStream os = null;

        model.setModelStats(modelStats);
        pmml.withModels(model);

        try {
            os = new FileOutputStream("test2.xml");
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);

        } catch (Exception e) {
            e.printStackTrace();
        }
                 */
    }


    private DataDictionary dict;
    private List<List<String>> rows;
    private List<List<String>> columns;

    @Test
    public void testOutputPMMLAsJSON() throws Exception {
        PMML pmml = PMMLUtils.loadPMML("src/test/resources/models/wdbc/Transform/model.xml");

        JSONUtils.writeValue(new File("test.json"), pmml);

    }

    private void loadData() throws Exception {


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
